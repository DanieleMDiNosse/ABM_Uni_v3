from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from abm_webapp.storage import SQLiteLiveSink


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalize_params_for_webapp(params: Dict[str, Any], *, run_root: Path) -> Dict[str, Any]:
    """
    Apply safe defaults/overrides for interactive web runs.

    Notes
    -----
    - `visualize=False` avoids generating many static artifacts during live use.
    - `verbose=False` disables tqdm + console prints but does NOT disable file logging
      (logging is only disabled by `light_mode=True`).
    """
    out = dict(params)
    out["results_root"] = run_root
    out["visualize"] = False
    out["liquidity_for_gif"] = False
    out["light_mode"] = False
    out["verbose"] = False
    # keep user-provided T/seed/etc. intact
    return out


def run_simulation_process(
    *,
    run_id: str,
    run_root: str,
    config_yaml: str,
    stop_event: Any,
    live_every: int = 5,
    log_flush_every: int = 200,
) -> None:
    """
    Entry point for the simulation worker process.

    Parameters
    ----------
    run_id
        Unique identifier for the run.
    run_root
        Output directory where `live.db` and logs are written.
    config_yaml
        YAML string containing a full `simulate:` mapping (same format as scenario files).
    stop_event
        A multiprocessing-compatible event used to request early termination.
    live_every
        Record live metrics every N blocks (smaller = more responsive, larger = lower overhead).
    log_flush_every
        Flush the simulation log buffer every N blocks so the UI can tail it live.

    Returns
    -------
    None

    Notes
    -----
    - This function must be defined at module top-level to be picklable on Windows.
    """
    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)

    db_path = run_root_path / "live.db"
    try:
        commit_every = max(1, int(live_every))
    except Exception:
        commit_every = 1
    sink = SQLiteLiveSink(
        db_path=db_path,
        run_id=str(run_id),
        params_yaml=str(config_yaml),
        results_root=run_root_path,
        T=int(_infer_T_from_yaml(config_yaml)),
        commit_every=commit_every,
    )

    config_path = run_root_path / "scenario.yml"
    _write_text(config_path, config_yaml)

    try:
        from run import simulate
        from utils import load_simulation_parameters

        scenario_label, params = load_simulation_parameters(config_path, simulate_func=simulate)
        params = _normalize_params_for_webapp(params, run_root=run_root_path)
        # The YAML loader populates defaults for all simulate() parameters, including
        # the live-streaming hooks. Remove them here to avoid passing duplicates.
        for k in ("live_sink", "live_every", "stop_event", "log_flush_every"):
            params.pop(k, None)

        # Run with live hooks enabled (implemented in run.py).
        simulate(
            **params,
            live_sink=sink,
            live_every=int(live_every),
            stop_event=stop_event,
            log_flush_every=int(log_flush_every),
        )
        # If the UI requested a stop, prefer marking the run as stopped.
        stopped = False
        try:
            is_set = getattr(stop_event, "is_set", None)
            stopped = bool(is_set()) if callable(is_set) else False
        except Exception:
            stopped = False
        sink.set_status(state="stopped" if stopped else "finished", message="stopped" if stopped else "completed")
    except Exception as exc:
        tb = traceback.format_exc()
        sink.set_status(state="error", message=f"{type(exc).__name__}: {exc}")
        _write_text(run_root_path / "error.log", tb)
    finally:
        try:
            sink.close()
        except Exception:
            # Avoid masking the original error; last-resort cleanup.
            pass


def _infer_T_from_yaml(config_yaml: str) -> int:
    """
    Best-effort extraction of `simulate.T` for UI progress reporting.

    Parameters
    ----------
    config_yaml
        Scenario YAML content.

    Returns
    -------
    int
        Requested number of blocks (defaults to 0 if missing/unparseable).
    """
    try:
        data = yaml.safe_load(config_yaml)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    sim = data.get("simulate")
    if not isinstance(sim, dict):
        return 0
    try:
        return int(sim.get("T", 0))
    except Exception:
        return 0
