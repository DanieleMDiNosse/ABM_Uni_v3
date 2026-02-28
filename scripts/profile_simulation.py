#!/usr/bin/env python3
"""Profile a simulation run (cProfile) and write reproducible artifacts."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for profiling a simulation.

    Parameters
    ----------
    argv
        Optional argument list. If None, reads from `sys.argv`.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.

    Notes
    -----
    - Defaults are intentionally short (small `T`) so profiling finishes quickly.

    Examples
    --------
    >>> args = parse_args([])  # doctest: +SKIP
    """
    p = argparse.ArgumentParser(description="Profile a simulation run with cProfile.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("abm_results/scenarios/test.yml"),
        help="Scenario YAML path (default: abm_results/scenarios/test.yml).",
    )
    p.add_argument("--T", type=int, default=2_000, help="Override simulation horizon (blocks).")
    p.add_argument("--skip-step", type=int, default=100, help="Override skip_step burn-in.")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed (default: use the YAML seed).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of functions to show in the top tables (default: 50).",
    )
    return p.parse_args(argv)


def _profile_simulation(params: Dict[str, Any], *, top_n: int) -> Tuple[cProfile.Profile, Dict[str, str]]:
    """Run a profiled simulation and return human-readable summaries.

    Parameters
    ----------
    params
        Simulation parameters mapping passed to `scripts.run.simulate`.
    top_n
        Number of functions to print in each top table.

    Returns
    -------
    tuple
        `(profiler, reports)` where `reports` is a mapping keyed by `"cumulative"`
        and `"tottime"`.

    Notes
    -----
    - This function intentionally avoids writing files so the caller controls
      output paths and overwrite behavior.

    Examples
    --------
    >>> isinstance(_profile_simulation({}, top_n=10), dict)  # doctest: +SKIP
    True
    """
    profiler = cProfile.Profile()
    profiler.enable()
    run_module.simulate(**params)
    profiler.disable()

    out: Dict[str, str] = {}
    for label, sort_key in (("cumulative", "cumulative"), ("tottime", "tottime")):
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_key)
        stats.print_stats(int(top_n))
        out[label] = stream.getvalue()

    return profiler, out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for profiling a simulation run.

    Parameters
    ----------
    argv
        Optional argument list. If None, reads from `sys.argv`.

    Returns
    -------
    int
        Process exit code (`0` on success).

    Notes
    -----
    - Writes outputs under:
      `abm_results/scenarios/<scenario_stem>/profile/<run_id>/`
      including `profile.prof`, `top_cumulative.txt`, `top_tottime.txt`,
      `metadata.json`, and `summary.csv`.

    Examples
    --------
    >>> main([])  # doctest: +SKIP
    0
    """
    run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]

    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    scenario_root = scenario_output_root(config_path)

    _, params = load_simulation_parameters(config_path, simulate_func=run_module.simulate)
    params = dict(params)

    params["T"] = int(args.T)
    params["skip_step"] = int(args.skip_step)
    params["light_mode"] = True
    params["visualize"] = False
    params["verbose"] = False
    if args.seed is not None:
        params["seed"] = int(args.seed)

    seed = int(params.get("seed", 0))
    run_id_base = safe_tag(f"profile_T{int(args.T)}_seed{seed}")
    out_root_base = scenario_root / "profile"
    out_root_base.mkdir(parents=True, exist_ok=True)
    out_root = make_unique_dir(out_root_base / run_id_base)

    snapshot_file(config_path, out_root / "config_snapshot.yml")
    manifest = build_run_manifest(script="profile_simulation", run_id=out_root.name, config_path=config_path)
    params["results_root"] = out_root
    write_json(
        out_root / "metadata.json",
        {
            **manifest.to_dict(),
            "seed": int(seed),
            "T": int(params.get("T", 0)),
            "skip_step": int(params.get("skip_step", 0)),
            "fee_mode": str(params.get("fee_mode", "")),
            "results_root": str(out_root),
        },
    )

    profiler, reports = _profile_simulation(params, top_n=int(args.top_n))

    prof_path = out_root / "profile.prof"
    profiler.dump_stats(str(prof_path))
    (out_root / "top_cumulative.txt").write_text(reports["cumulative"], encoding="utf-8")
    (out_root / "top_tottime.txt").write_text(reports["tottime"], encoding="utf-8")

    write_csv_rows(
        out_root / "summary.csv",
        [
            {
                "run_id": out_root.name,
                "config_path": str(config_path),
                "seed": int(seed),
                "T": int(params.get("T", 0)),
                "skip_step": int(params.get("skip_step", 0)),
                "light_mode": bool(params.get("light_mode", False)),
            }
        ],
    )

    print(f"[profile] wrote: {out_root}")
    print(f"[profile] stats: {prof_path}")
    print(f"[profile] top:   {out_root / 'top_cumulative.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
