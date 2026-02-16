from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from abm_webapp.storage import SQLiteLiveSink

# Heartbeat interval in seconds – the worker updates its DB heartbeat
# at least this often so the UI can detect crashes.
_HEARTBEAT_INTERVAL_S: float = 10.0


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


def _start_heartbeat_thread(
    sink: SQLiteLiveSink,
    stop_event: Any,
    *,
    interval: float = _HEARTBEAT_INTERVAL_S,
) -> threading.Thread:
    """
    Launch a daemon thread that periodically updates the heartbeat timestamp
    in the run's DB so that the UI can detect a dead worker.
    """

    def _loop() -> None:
        while True:
            # Check if the stop event is set
            try:
                is_set = getattr(stop_event, "is_set", None)
                if callable(is_set) and is_set():
                    break
            except Exception:
                pass
            try:
                sink.update_heartbeat()
            except Exception:
                pass
            # Use the stop_event's wait (with timeout) for a cleaner exit
            try:
                wait_fn = getattr(stop_event, "wait", None)
                if callable(wait_fn):
                    wait_fn(timeout=interval)
                else:
                    import time
                    time.sleep(interval)
            except Exception:
                import time
                time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="heartbeat")
    t.start()
    return t


def run_simulation_process(
    *,
    run_id: str,
    run_root: str,
    config_yaml: str,
    stop_event: Any,
    live_every: int = 5,
    log_flush_every: int = 200,
    meta_json: Optional[str] = None,
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
    meta_json
        Optional JSON string with extended run metadata for reproducibility.

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
        meta_json=meta_json,
    )

    # Persist the validated config snapshot (canonical form)
    config_path = run_root_path / "scenario.yml"
    _write_text(config_path, config_yaml)

    # Persist run metadata as a separate JSON file for easy inspection
    if meta_json:
        _write_text(run_root_path / "run_meta.json", meta_json)

    # Start the heartbeat thread
    hb_thread = _start_heartbeat_thread(sink, stop_event)

    try:
        from scripts.run import simulate
        from core.utils import load_simulation_parameters

        scenario_label, params = load_simulation_parameters(config_path, simulate_func=simulate)
        params = _normalize_params_for_webapp(params, run_root=run_root_path)
        # The YAML loader populates defaults for all simulate() parameters, including
        # the live-streaming hooks. Remove them here to avoid passing duplicates.
        for k in ("live_sink", "live_every", "stop_event", "log_flush_every"):
            params.pop(k, None)

        # Run with live hooks enabled (implemented in scripts/run.py).
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
        if stopped:
            sink.set_status(state="stopped", message="stopped", stop_reason="user_stop")
        else:
            sink.set_status(state="finished", message="completed")
    except Exception as exc:
        tb = traceback.format_exc()
        sink.set_status(state="error", message=f"{type(exc).__name__}: {exc}", stop_reason="exception")
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
