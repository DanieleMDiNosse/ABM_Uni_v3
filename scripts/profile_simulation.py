#!/usr/bin/env python3
"""Profile a simulation run: CPU (cProfile), memory (tracemalloc), disk & I/O.

Designed to pinpoint bottlenecks when scaling from 10k → 500k steps.
Outputs a structured report with extrapolations for large T.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run as run_module
from core.artifacts import build_run_manifest, make_unique_dir, safe_tag, snapshot_file, write_csv_rows, write_json
from core.utils import load_simulation_parameters, scenario_output_root


def _silent_tqdm(iterable=None, **kwargs):
    """Silence tqdm inside `scripts.run.simulate` to keep profiling output clean."""
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


def _dir_size_bytes(path: Path) -> int:
    """Recursively compute total size of all files under *path*."""
    total = 0
    if path.is_file():
        return path.stat().st_size
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for CLI arguments."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer for CLI arguments."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _estimate_result_payload(result: Any) -> Tuple[int, int]:
    """Estimate retained payload size from the simulation return mapping.

    Parameters
    ----------
    result
        Return value from ``scripts.run.simulate``.

    Returns
    -------
    tuple[int, int]
        Number of top-level keys and an approximate count of retained elements
        across list-like and array-like outputs.

    Notes
    -----
    - This intentionally focuses on horizon-scaling containers.
    - Array-like objects contribute their numeric ``size`` when available.
    """
    if not isinstance(result, dict):
        return 0, 0

    element_count = 0
    for value in result.values():
        if isinstance(value, (list, tuple)):
            element_count += len(value)
            continue

        size = getattr(value, "size", None)
        if size is not None and not callable(size):
            try:
                element_count += int(size)
                continue
            except (TypeError, ValueError, OverflowError):
                pass

        if isinstance(value, dict):
            element_count += len(value)

    return len(result), element_count


def _format_profile_function(filename: str, lineno: int, func_name: str) -> str:
    """Return a compact ``path:line:function`` label for profile summaries."""
    try:
        resolved = Path(filename).resolve()
        display = resolved.relative_to(_REPO_ROOT)
    except Exception:
        display = Path(filename).name or filename
    return f"{display}:{lineno}:{func_name}"


def _top_profile_lines(profiler: cProfile.Profile, *, sort_key: str, top_n: int) -> List[str]:
    """Render a compact top-N summary from ``pstats``."""
    stats = pstats.Stats(profiler).sort_stats(sort_key)
    rows: List[str] = []
    for func in stats.fcn_list[:top_n]:
        cc, nc, tt, ct, _ = stats.stats[func]
        filename, lineno, func_name = func
        rows.append(
            "  "
            f"{_format_profile_function(filename, lineno, func_name)}  "
            f"cum={ct:.3f}s self={tt:.3f}s calls={nc} primitive_calls={cc}"
        )
    if not rows:
        rows.append("  No cProfile rows were captured.")
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile a simulation run: CPU, memory, disk usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.profile_simulation --T 2000\n"
            "  python -m scripts.profile_simulation --T 10000 --light-mode\n"
            "  python -m scripts.profile_simulation --T 5000 --full-mode\n"
        ),
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("abm_results/scenarios/test.yml"),
        help="Scenario YAML path (default: abm_results/scenarios/test.yml).",
    )
    p.add_argument("--T", type=_positive_int, default=2_000, help="Override simulation horizon (blocks).")
    p.add_argument("--skip-step", type=_nonnegative_int, default=100, help="Override skip_step burn-in.")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    p.add_argument("--top-n", type=_positive_int, default=50, help="Top N functions in cProfile tables.")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--full-mode", action="store_true",
        help="Run with light_mode=False (verbose log + plots). "
             "Default is light_mode=True for faster profiling.",
    )
    mode_group.add_argument(
        "--light-mode",
        action="store_true",
        help="Explicitly force light_mode=True. This matches the default and helps scripted runs stay explicit.",
    )
    p.add_argument(
        "--extrapolate-to", type=_positive_int, default=500_000,
        help="Target T for linear extrapolation (default: 500000).",
    )
    return p.parse_args(argv)


def _profile_simulation(
    params: Dict[str, Any], *, top_n: int
) -> Tuple[cProfile.Profile, Dict[str, str], Dict[str, Any]]:
    """Run a profiled simulation with CPU + memory instrumentation.

    Returns (profiler, cprofile_reports, metrics) where metrics includes
    wall time, peak RSS, tracemalloc stats, and the return dict size estimate.
    """
    # Start tracemalloc for memory profiling
    tracemalloc.start(25)  # 25 frames of traceback for attribution

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    result = run_module.simulate(**params)
    profiler.disable()
    wall_seconds = time.perf_counter() - t0

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    snapshot = tracemalloc.take_snapshot()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # cProfile text reports
    reports: Dict[str, str] = {}
    for label, sort_key in (("cumulative", "cumulative"), ("tottime", "tottime")):
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_key)
        stats.print_stats(int(top_n))
        reports[label] = stream.getvalue()

    # Top memory allocators (by size)
    top_mem_stats = snapshot.statistics("lineno")
    mem_lines: List[str] = []
    for stat in top_mem_stats[:30]:
        mem_lines.append(f"  {_fmt_bytes(stat.size):>12s}  {stat.count:>8d} blocks  {stat.traceback}")

    # Estimate retained payload size without keeping the full result alive longer
    # than needed after the profiled section.
    return_dict_keys, return_dict_elements = _estimate_result_payload(result)
    del result

    metrics = {
        "wall_seconds": wall_seconds,
        "rss_before_kib": rss_before,
        "rss_after_kib": rss_after,
        "peak_rss_kib": rss_after,  # ru_maxrss is high-water mark
        "tracemalloc_current_bytes": current_mem,
        "tracemalloc_peak_bytes": peak_mem,
        "top_memory_lines": mem_lines,
        "return_dict_keys": return_dict_keys,
        "return_dict_elements": return_dict_elements,
    }
    return profiler, reports, metrics


def _build_report(
    *,
    T: int,
    block_time: int,
    light_mode: bool,
    wall_seconds: float,
    rss_peak_kib: int,
    tracemalloc_peak: int,
    tracemalloc_current: int,
    return_dict_keys: int,
    return_dict_elements: int,
    disk_bytes: int,
    top_memory_lines: List[str],
    top_cumulative_lines: List[str],
    top_tottime_lines: List[str],
    extrapolate_to: int,
) -> str:
    """Build the human-readable profiling summary report."""
    ratio = extrapolate_to / max(T, 1)
    micro_steps_per_step = max(block_time - 1, 1)
    projected_return_bytes = int(return_dict_elements * 8 * ratio)
    projected_micro_bytes = int(micro_steps_per_step * extrapolate_to * 3 * 8)

    lines = [
        "=" * 72,
        "  SIMULATION PROFILING REPORT",
        "=" * 72,
        "",
        f"  T (steps)         : {T:,}",
        f"  block_time        : {block_time}",
        f"  light_mode        : {light_mode}",
        f"  micro steps/block : {micro_steps_per_step}",
        f"  total micro steps : ~{T * micro_steps_per_step:,}",
        "",
        "--- WALL CLOCK ---",
        f"  Total             : {wall_seconds:.2f}s",
        f"  Per step          : {wall_seconds / max(T, 1) * 1000:.3f} ms/step",
        f"  Extrapolated T={extrapolate_to:,}: ~{wall_seconds * ratio:.0f}s "
        f"({wall_seconds * ratio / 60:.1f} min, {wall_seconds * ratio / 3600:.2f} hr)",
        "",
        "--- MEMORY ---",
        f"  Peak RSS          : {_fmt_bytes(rss_peak_kib * 1024)}",
        f"  tracemalloc peak  : {_fmt_bytes(tracemalloc_peak)}",
        f"  tracemalloc live  : {_fmt_bytes(tracemalloc_current)}",
        f"  Return dict keys  : {return_dict_keys}",
        f"  Return dict elems : {return_dict_elements:,}",
        f"  Est. return dict  : ~{_fmt_bytes(return_dict_elements * 8)} (float64)",
        f"  Extrapolated T={extrapolate_to:,}: ~{_fmt_bytes(projected_return_bytes)} return dict alone",
        "",
        f"  KEY MEMORY CONCERN at T={extrapolate_to:,}:",
        f"    ~80 Python lists × {extrapolate_to:,} floats × 8 bytes ≈ "
        f"{_fmt_bytes(80 * extrapolate_to * 8)}",
        f"    + micro_steps lists: ~{micro_steps_per_step} × {extrapolate_to:,} × 3 lists × 8 bytes ≈ "
        f"{_fmt_bytes(projected_micro_bytes)}",
        "",
        "--- DISK I/O ---",
        f"  Output dir size   : {_fmt_bytes(disk_bytes)}",
        f"  Extrapolated T={extrapolate_to:,}: ~{_fmt_bytes(int(disk_bytes * ratio))}",
        "",
        "--- HOT FUNCTIONS (cumulative time) ---",
    ]
    lines.extend(top_cumulative_lines)
    lines.extend([
        "",
        "--- HOT FUNCTIONS (self time) ---",
    ])
    lines.extend(top_tottime_lines)
    lines.extend([
        "",
        "--- TOP MEMORY ALLOCATORS (tracemalloc, by size) ---",
    ])
    lines.extend(top_memory_lines[:20])
    lines.extend([
        "",
        "--- SCALING NOTES ---",
        "  Compare this report across multiple T values to detect superlinear growth.",
        "  Use `top_tottime.txt` to identify hot inner loops and `top_cumulative.txt`",
        "  to identify expensive call chains.",
        "",
        f"  The profiled payload already retains about {_fmt_bytes(return_dict_elements * 8)}",
        f"  of numeric series data; under linear growth that projects to",
        f"  {_fmt_bytes(projected_return_bytes)} at T={extrapolate_to:,}.",
        "",
        f"  The three core micro-step series alone project to about {_fmt_bytes(projected_micro_bytes)}",
        f"  at T={extrapolate_to:,}.",
        "",
        (
            "  This run used full mode, so wall-clock time includes plotting and file"
            " generation overhead in addition to core simulation cost."
        )
        if not light_mode
        else (
            "  This run used light mode, so plotting and most artifact generation were"
            " disabled. This is the better baseline for core-engine profiling."
        ),
        "",
        (
            "  If output size is already large in full mode, keep `--light-mode` for"
            " scaling studies and reserve `--full-mode` for artifact-heavy diagnostics."
        )
        if not light_mode
        else (
            "  If you need to profile plotting and export overhead explicitly, rerun"
            " the same configuration with `--full-mode` and compare the two reports."
        ),
        "",
        "=" * 72,
    ])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint: profile simulation with CPU + memory + disk analysis."""
    run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]

    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    scenario_root = scenario_output_root(config_path)

    _, params = load_simulation_parameters(config_path, simulate_func=run_module.simulate)
    params = dict(params)

    T = int(args.T)
    params["T"] = T
    params["skip_step"] = int(args.skip_step)
    light_mode = True
    if args.full_mode:
        light_mode = False
    if args.light_mode:
        light_mode = True
    params["light_mode"] = light_mode
    params["visualize"] = not light_mode  # only visualize in full mode
    params["verbose"] = False  # always silence tqdm/prints during profiling
    if args.seed is not None:
        params["seed"] = int(args.seed)

    seed = int(params.get("seed", 0))
    block_time = int(params.get("block_time", 12))
    mode_tag = "full" if not light_mode else "light"
    run_id_base = safe_tag(f"profile_T{T}_{mode_tag}_seed{seed}")
    out_root_base = scenario_root / "profile"
    out_root_base.mkdir(parents=True, exist_ok=True)
    out_root = make_unique_dir(out_root_base / run_id_base)

    snapshot_file(config_path, out_root / "config_snapshot.yml")
    manifest = build_run_manifest(
        script="profile_simulation", run_id=out_root.name, config_path=config_path
    )
    params["results_root"] = out_root
    write_json(
        out_root / "metadata.json",
        {
            **manifest.to_dict(),
            "seed": seed,
            "T": T,
            "skip_step": params["skip_step"],
            "fee_mode": str(params.get("fee_mode", "")),
            "light_mode": light_mode,
            "results_root": str(out_root),
        },
    )

    print(f"[profile] Running T={T:,}, light_mode={light_mode}, seed={seed}")
    print(f"[profile] Output: {out_root}")
    print()

    # --- Run profiled simulation ---
    profiler, reports, metrics = _profile_simulation(params, top_n=int(args.top_n))

    # --- Measure disk usage of output directory ---
    disk_bytes = _dir_size_bytes(out_root)

    # --- Save cProfile artifacts ---
    prof_path = out_root / "profile.prof"
    profiler.dump_stats(str(prof_path))
    (out_root / "top_cumulative.txt").write_text(reports["cumulative"], encoding="utf-8")
    (out_root / "top_tottime.txt").write_text(reports["tottime"], encoding="utf-8")
    top_cumulative_lines = _top_profile_lines(
        profiler,
        sort_key="cumulative",
        top_n=min(int(args.top_n), 10),
    )
    top_tottime_lines = _top_profile_lines(
        profiler,
        sort_key="tottime",
        top_n=min(int(args.top_n), 10),
    )

    # --- Build and save the comprehensive report ---
    report = _build_report(
        T=T,
        block_time=block_time,
        light_mode=light_mode,
        wall_seconds=metrics["wall_seconds"],
        rss_peak_kib=metrics["peak_rss_kib"],
        tracemalloc_peak=metrics["tracemalloc_peak_bytes"],
        tracemalloc_current=metrics["tracemalloc_current_bytes"],
        return_dict_keys=metrics["return_dict_keys"],
        return_dict_elements=metrics["return_dict_elements"],
        disk_bytes=disk_bytes,
        top_memory_lines=metrics["top_memory_lines"],
        top_cumulative_lines=top_cumulative_lines,
        top_tottime_lines=top_tottime_lines,
        extrapolate_to=int(args.extrapolate_to),
    )
    (out_root / "profile_report.txt").write_text(report, encoding="utf-8")

    # Print the report to stdout
    print(report)
    print()

    # Print top 20 cumulative cProfile entries
    print("--- TOP 20 cProfile (cumulative time) ---")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())

    # --- Save summary CSV ---
    write_csv_rows(
        out_root / "summary.csv",
        [
            {
                "run_id": out_root.name,
                "config_path": str(config_path),
                "seed": seed,
                "T": T,
                "skip_step": params["skip_step"],
                "light_mode": light_mode,
                "wall_seconds": round(metrics["wall_seconds"], 3),
                "peak_rss_mib": round(metrics["peak_rss_kib"] / 1024, 1),
                "tracemalloc_peak_mib": round(metrics["tracemalloc_peak_bytes"] / (1024 * 1024), 1),
                "disk_bytes": disk_bytes,
                "return_dict_elements": metrics["return_dict_elements"],
            }
        ],
    )

    print(f"[profile] Artifacts saved to: {out_root}")
    print(f"[profile]   profile.prof          – load with snakeviz or pstats")
    print(f"[profile]   profile_report.txt    – profile summary with scaling notes")
    print(f"[profile]   top_cumulative.txt    – cProfile top {args.top_n} (cumtime)")
    print(f"[profile]   top_tottime.txt       – cProfile top {args.top_n} (tottime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
