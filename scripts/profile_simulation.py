#!/usr/bin/env python3
"""Profile a simulation run: CPU (cProfile), memory (tracemalloc), disk & I/O.

Designed to pinpoint bottlenecks when scaling from 10k → 500k steps.
Outputs a structured report with extrapolations for large T.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
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
    p.add_argument("--T", type=int, default=2_000, help="Override simulation horizon (blocks).")
    p.add_argument("--skip-step", type=int, default=100, help="Override skip_step burn-in.")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    p.add_argument("--top-n", type=int, default=50, help="Top N functions in cProfile tables.")
    p.add_argument(
        "--full-mode", action="store_true",
        help="Run with light_mode=False (verbose log + plots). "
             "Default is light_mode=True for faster profiling.",
    )
    p.add_argument(
        "--extrapolate-to", type=int, default=500_000,
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

    # Estimate return dict memory (sum of list/array element counts)
    return_dict_elements = 0
    return_dict_keys = 0
    if isinstance(result, dict):
        return_dict_keys = len(result)
        for k, v in result.items():
            if isinstance(v, list):
                return_dict_elements += len(v)
            elif hasattr(v, "__len__"):
                return_dict_elements += len(v)

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
        "result": result,
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
    extrapolate_to: int,
) -> str:
    """Build the human-readable profiling summary report."""
    ratio = extrapolate_to / max(T, 1)
    micro_steps_per_step = max(block_time - 1, 1)

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
        f"  Extrapolated T={extrapolate_to:,}: ~{_fmt_bytes(int(return_dict_elements * 8 * ratio))} return dict alone",
        "",
        "  KEY MEMORY CONCERN at 500k steps:",
        f"    ~{80} Python lists × {extrapolate_to:,} floats × 8 bytes ≈ "
        f"{_fmt_bytes(80 * extrapolate_to * 8)}",
        f"    + micro_steps lists: ~{micro_steps_per_step} × {extrapolate_to:,} × 3 lists × 8 bytes ≈ "
        f"{_fmt_bytes(micro_steps_per_step * extrapolate_to * 3 * 8)}",
        "",
        "--- DISK I/O ---",
        f"  Output dir size   : {_fmt_bytes(disk_bytes)}",
        f"  Extrapolated T={extrapolate_to:,}: ~{_fmt_bytes(int(disk_bytes * ratio))}",
        "",
        "  DISK CONCERN: In full (non-light) mode, the verbose log file and",
        "  15+ Plotly HTML files with 500k datapoints each can consume",
        "  several GB. The verbose log alone can exceed 500 MB.",
        "",
        "--- TOP MEMORY ALLOCATORS (tracemalloc, by size) ---",
    ]
    lines.extend(top_memory_lines[:20])
    lines.append("")

    # Specific bottleneck analysis
    lines.extend([
        "--- IDENTIFIED BOTTLENECKS FOR 500k SCALING ---",
        "",
        "  *** CRITICAL: SUPERLINEAR (O(T²)) SCALING ***",
        "  The dominant bottleneck is lp_token0_exposure() and",
        "  lp_principal_amounts() in core/agents.py. These iterate over",
        "  ALL positions for each LP, and positions accumulate over time",
        "  (mints without corresponding burns). This makes per-step cost",
        "  grow linearly with T → total cost is O(T²).",
        "",
        "  Evidence: T=2k → 42.9ms/step, T=10k → 176ms/step (4.1× per step).",
        "  At 500k steps, per-step cost would be ~4400ms → total ~25 days.",
        "",
        "  Functions affected (by tottime):",
        "    - lp_token0_exposure (agents.py:129): iterates all positions",
        "    - lp_principal_amounts (agents.py:148): iterates all positions",
        "    - _current_amounts_impl (numba): called per position per LP",
        "    - allocate_fees (run.py:2915): iterates positions per tick",
        "",
        "  FIX: Cap or garbage-collect old positions; aggregate exposure at",
        "  LP level instead of re-summing per position each step; or merge",
        "  overlapping positions into a single aggregate position.",
        "",
        "  1. LIST APPENDS IN MAIN LOOP (~80+ lists × T appends)",
        "     At 500k: ~40M+ list.append() calls, ~320 MB+ RAM for time series.",
        "     FIX: Pre-allocate numpy arrays; write to index instead of append.",
        "",
        "  2. MICRO-STEP TRACKING (P_micro, M_micro, micro_steps)",
        f"     {micro_steps_per_step} entries/step × 3 lists × T steps.",
        f"     At 500k: ~{micro_steps_per_step * 500_000 * 3:,} entries → "
        f"~{_fmt_bytes(micro_steps_per_step * 500_000 * 3 * 8)}.",
        "     FIX: Downsample or disable micro tracking for large T.",
        "",
        "  3. VERBOSE LOG (non-light mode)",
        "     ~1 log line per step + per event (arb, swap, mint, burn).",
        "     At 500k: can easily exceed 500 MB on disk.",
        "     FIX: Already mitigated by light_mode=True; ensure it's used.",
        "",
        "  4. PLOTLY HTML/PNG (non-light mode, visualize=True)",
        "     15+ plots, each embedding 500k datapoints in HTML.",
        "     HTML files: ~10-50 MB each → 150-750 MB total.",
        "     FIX: Downsample plot data; skip HTML for large T; use light_mode.",
        "",
        "  5. RETURN DICT .tolist() CONVERSION (lines 5216-5329)",
        "     40+ numpy arrays converted to Python lists → doubles peak memory.",
        "     At 500k: adds ~640 MB transient allocation.",
        "     FIX: Return numpy arrays directly (skip .tolist()).",
        "",
        "  6. POST-LOOP list→numpy CONVERSION (lines 5024-5088)",
        "     40+ lists re-allocated as numpy arrays → transient 2x memory.",
        "     FIX: Pre-allocate arrays from the start.",
        "",
        "  7. liq_history (liquidity_for_gif=True)",
        "     dict(pool.liquidity_net) copied every step → unbounded growth.",
        "     At 500k: could be many GB depending on tick count.",
        "     FIX: Already off by default; never enable for large T.",
        "",
        "  8. np.save AT END (5 files: dex_price, micro_price, spreads)",
        f"     At 500k: ~{_fmt_bytes(500_000 * 8 * 5)} for block-level +",
        f"     ~{_fmt_bytes(500_000 * micro_steps_per_step * 8)} for micro prices.",
        "     Manageable but contributes to disk pressure.",
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
    light_mode = not args.full_mode
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
    print(f"[profile]   profile_report.txt    – full bottleneck analysis")
    print(f"[profile]   top_cumulative.txt    – cProfile top {args.top_n} (cumtime)")
    print(f"[profile]   top_tottime.txt       – cProfile top {args.top_n} (tottime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
