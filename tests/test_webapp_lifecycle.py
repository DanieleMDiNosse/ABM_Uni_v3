"""
Webapp lifecycle, crash recovery, and config validation tests.

All tests use temporary directories and run fast (<30s total).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from multiprocessing import Event
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from abm_webapp.storage import (
    SCHEMA_VERSION,
    SQLiteLiveSink,
    _connect_sqlite,
    _ensure_schema,
    _utc_now_iso,
    is_heartbeat_stale,
    read_metrics,
    read_schema_version,
    read_status,
    scan_and_recover,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_step_row(t: int, **overrides: Any) -> Dict[str, Any]:
    """Build a minimal valid metrics row."""
    base = dict(
        t=t,
        dex_price=1.0 + 0.001 * t,
        cex_price=1.0 + 0.0012 * t,
        cex_sigma=0.0001,
        band_lo=0.99,
        band_hi=1.01,
        sr_pnl_step=0.0,
        noise_pnl_step=0.0,
        arb_pnl_step=0.0,
        lp_pnl_active=0.0,
        lp_pnl_passive=0.0,
        lp_unhedged_active=0.0,
        lp_unhedged_passive=0.0,
        lp_fee_value_total=0.0,
        lp_lvr_total=0.0,
        jiter_pnl=0.0,
        dex_notional_y=10.0,
        d_lvr_total=0.001,
        d_fee_value_total=0.002,
        trader_exec_count=1,
        arb_exec_count=0,
        sr_exec_count=1,
        noise_exec_count=0,
        sr_cex_exec_count=0,
        sr_dex_exec_count=1,
        fee=0.003,
        fee_sigma=0.0,
        fee_basis_ticks=0.0,
        fee_signal=0.0,
    )
    base.update(overrides)
    return base


_MINIMAL_YAML = """\
fee_mode: static
simulate:
  config_name: test
  block_time: 5
  T: 5
  seed: 42
  cex_mu: 0.0
  cex_sigma: 0.0001
"""


# ---------------------------------------------------------------------------
# 1. Smoke test: short run lifecycle via storage layer
# ---------------------------------------------------------------------------
class TestSmokeLifecycle:
    """Verify that a short run writes DB correctly and transitions to completed."""

    def test_sink_writes_metrics_and_completes(self, tmp_path: Path) -> None:
        db_path = tmp_path / "live.db"
        meta = json.dumps({"app_version": "0.2.0", "seed": 42})
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="smoke_run",
            params_yaml=_MINIMAL_YAML,
            results_root=tmp_path,
            T=5,
            commit_every=2,
            meta_json=meta,
        )

        # Verify running state
        status = read_status(db_path)
        assert status is not None
        assert status.state == "running"
        assert status.pid == os.getpid()
        assert status.heartbeat_at is not None

        # Record a few steps
        for t in range(5):
            sink.record_step(_make_step_row(t))
        sink.set_status(state="finished", message="completed")
        sink.close()

        # Verify final state
        status = read_status(db_path)
        assert status is not None
        assert status.state == "finished"
        assert status.t_last == 4
        assert status.stopped_at is not None

        rows = read_metrics(db_path)
        assert len(rows) == 5
        assert rows[0]["t"] == 0
        assert rows[-1]["t"] == 4
        assert rows[-1]["dex_price"] == pytest.approx(1.004, abs=0.001)

    def test_schema_version_is_set(self, tmp_path: Path) -> None:
        db_path = tmp_path / "live.db"
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="sv_run",
            params_yaml=_MINIMAL_YAML,
            results_root=tmp_path,
            T=1,
        )
        sink.close()
        assert read_schema_version(db_path) == SCHEMA_VERSION

    def test_heartbeat_update(self, tmp_path: Path) -> None:
        db_path = tmp_path / "live.db"
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="hb_run",
            params_yaml=_MINIMAL_YAML,
            results_root=tmp_path,
            T=1,
        )
        # Initial heartbeat
        status1 = read_status(db_path)
        assert status1 is not None
        hb1 = status1.heartbeat_at

        time.sleep(0.05)
        sink.update_heartbeat()

        status2 = read_status(db_path)
        assert status2 is not None
        assert status2.heartbeat_at is not None
        assert status2.heartbeat_at >= hb1  # type: ignore[operator]
        sink.close()


# ---------------------------------------------------------------------------
# 2. Crash recovery tests
# ---------------------------------------------------------------------------
class TestCrashRecovery:
    """Simulate crashed/stale workers and verify startup scan marks them abandoned."""

    def _create_running_db(
        self,
        db_path: Path,
        *,
        run_id: str = "crash_run",
        pid: int = 999999999,
        heartbeat_at: str = "",
    ) -> None:
        """Create a DB that looks like a running worker (but isn't)."""
        conn = _connect_sqlite(db_path)
        _ensure_schema(conn)
        if not heartbeat_at:
            # Default to a stale heartbeat (2 minutes ago)
            stale_time = datetime.now(timezone.utc) - timedelta(minutes=2)
            heartbeat_at = stale_time.isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT OR REPLACE INTO run_meta(run_id, created_at, params_yaml, results_root, T)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, _utc_now_iso(), _MINIMAL_YAML, str(db_path.parent), 100),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO run_status(
                run_id, state, t_last, message, updated_at, log_path, pid, heartbeat_at
            )
            VALUES (?, 'running', 50, '', ?, '', ?, ?)
            """,
            (run_id, _utc_now_iso(), pid, heartbeat_at),
        )
        conn.commit()
        conn.close()

    def test_dead_pid_marked_abandoned(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "web_runs" / "dead_pid_run"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "live.db"
        self._create_running_db(db_path, run_id="dead_pid_run", pid=999999999)

        # Verify it's initially running
        status = read_status(db_path)
        assert status is not None
        assert status.state == "running"

        # Scan should mark it abandoned (PID 999999999 shouldn't exist)
        recovered = scan_and_recover(tmp_path / "web_runs")
        assert "dead_pid_run" in recovered

        status = read_status(db_path)
        assert status is not None
        assert status.state == "abandoned"
        assert "PID" in status.message
        assert status.stop_reason == "crash_recovery"

    def test_stale_heartbeat_marked_abandoned(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "web_runs" / "stale_hb_run"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "live.db"

        # Use current PID (alive) but with a stale heartbeat
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        self._create_running_db(
            db_path,
            run_id="stale_hb_run",
            pid=os.getpid(),  # alive!
            heartbeat_at=stale_time.isoformat(timespec="seconds"),
        )

        recovered = scan_and_recover(tmp_path / "web_runs")
        assert "stale_hb_run" in recovered

        status = read_status(db_path)
        assert status is not None
        assert status.state == "abandoned"
        assert "heartbeat stale" in status.message

    def test_finished_run_not_touched(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "web_runs" / "done_run"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "live.db"
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="done_run",
            params_yaml=_MINIMAL_YAML,
            results_root=run_dir,
            T=1,
        )
        sink.set_status(state="finished", message="ok")
        sink.close()

        recovered = scan_and_recover(tmp_path / "web_runs")
        assert recovered == []

        status = read_status(db_path)
        assert status is not None
        assert status.state == "finished"

    def test_empty_web_runs_dir(self, tmp_path: Path) -> None:
        recovered = scan_and_recover(tmp_path / "nonexistent")
        assert recovered == []

    def test_is_heartbeat_stale_edge_cases(self) -> None:
        assert is_heartbeat_stale(None) is True
        assert is_heartbeat_stale("not-a-date") is True
        fresh = datetime.now(timezone.utc).isoformat(timespec="seconds")
        assert is_heartbeat_stale(fresh, threshold_s=60.0) is False
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
        assert is_heartbeat_stale(old, threshold_s=60.0) is True


# ---------------------------------------------------------------------------
# 3. Config validation tests
# ---------------------------------------------------------------------------
class TestConfigValidation:
    """Verify that invalid configs are rejected with friendly errors."""

    def test_empty_yaml_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        ok, err = validate_scenario_text("")
        assert not ok
        assert "empty" in err.lower()

    def test_missing_simulate_section_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        ok, err = validate_scenario_text("fee_mode: static\n")
        assert not ok
        assert "simulate" in err.lower()

    def test_unknown_top_level_keys_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        yaml_text = "fee_mode: static\nsimulate:\n  T: 100\nextra_junk: true\n"
        ok, err = validate_scenario_text(yaml_text)
        assert not ok
        assert "extra_junk" in err

    def test_invalid_fee_mode_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        yaml_text = "fee_mode: magic_mode\nsimulate:\n  T: 100\n  seed: 1\n  block_time: 5\n  cex_mu: 0.0\n  cex_sigma: 0.0001\n"
        ok, err = validate_scenario_text(yaml_text)
        assert not ok
        assert "magic_mode" in err

    def test_T_exceeds_max_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        yaml_text = f"fee_mode: static\nsimulate:\n  T: 99999999\n  seed: 1\n  block_time: 5\n  cex_mu: 0.0\n  cex_sigma: 0.0001\n"
        ok, err = validate_scenario_text(yaml_text)
        assert not ok
        assert "exceeds" in err.lower() or "maximum" in err.lower()

    def test_negative_cex_sigma_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        yaml_text = "fee_mode: static\nsimulate:\n  T: 100\n  seed: 1\n  block_time: 5\n  cex_mu: 0.0\n  cex_sigma: -0.001\n"
        ok, err = validate_scenario_text(yaml_text)
        assert not ok
        assert "below minimum" in err.lower()

    def test_conflicting_fee_modes_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        yaml_text = "fee_mode: static\nsimulate:\n  fee_mode: toxicity\n  T: 100\n  seed: 1\n  block_time: 5\n  cex_mu: 0.0\n  cex_sigma: 0.0001\n"
        ok, err = validate_scenario_text(yaml_text)
        assert not ok
        assert "conflicting" in err.lower()

    def test_valid_config_accepted(self) -> None:
        from abm_webapp.config import validate_scenario_text

        ok, err = validate_scenario_text(_MINIMAL_YAML)
        assert ok, f"Expected valid, got error: {err}"
        assert err == ""

    def test_oversized_yaml_rejected(self) -> None:
        from abm_webapp.config import safe_load_yaml

        big = "a: " + "x" * 600_000
        data, err = safe_load_yaml(big)
        assert data is None
        assert "limit" in err.lower()

    def test_canonical_yaml_output(self) -> None:
        from abm_webapp.config import validate_scenario, canonical_yaml

        result, err = validate_scenario(_MINIMAL_YAML)
        assert result is not None
        canon = canonical_yaml(result)
        assert "fee_mode" in canon
        assert "simulate" in canon

    def test_block_time_too_low_rejected(self) -> None:
        from abm_webapp.config import validate_scenario_text

        yaml_text = "fee_mode: static\nsimulate:\n  T: 100\n  seed: 1\n  block_time: 1\n  cex_mu: 0.0\n  cex_sigma: 0.0001\n"
        ok, err = validate_scenario_text(yaml_text)
        assert not ok
        assert "below minimum" in err.lower()

    def test_repo_scenarios_pass_full_webapp_validation(self) -> None:
        from abm_webapp.app import _validate_config_against_simulate

        for scenario_name in ("test.yml", "vol_conditioned_wide.yml"):
            yaml_text = (Path("abm_results/scenarios") / scenario_name).read_text(encoding="utf-8")
            ok, err = _validate_config_against_simulate(yaml_text)
            assert ok, f"{scenario_name} should be runnable in the webapp, got: {err}"

    def test_list_scenario_files_filters_non_runnable_yaml(self, tmp_path: Path) -> None:
        from abm_webapp.app import _list_scenario_files, _partition_scenario_files

        runnable_text = (Path("abm_results/scenarios") / "test.yml").read_text(encoding="utf-8")
        (tmp_path / "valid.yml").write_text(runnable_text, encoding="utf-8")
        (tmp_path / "sweep.yml").write_text(
            "version: 1\nname: sweep\nfee_mode: static\nsweeps: {}\n",
            encoding="utf-8",
        )
        (tmp_path / "broken.yml").write_text("simulate: [\n", encoding="utf-8")

        scenario_files = _list_scenario_files(tmp_path)
        assert [path.name for path in scenario_files] == ["valid.yml"]

        _, rejected = _partition_scenario_files(tmp_path)
        assert "sweep.yml" in rejected
        assert "broken.yml" in rejected


# ---------------------------------------------------------------------------
# 4. Run metadata tests
# ---------------------------------------------------------------------------
class TestRunMeta:
    """Verify run metadata collection."""

    def test_collect_run_meta_fields(self) -> None:
        from abm_webapp.run_meta import collect_run_meta

        meta = collect_run_meta(
            app_version="0.2.0",
            seed=42,
            config_yaml=_MINIMAL_YAML,
        )
        assert meta["app_version"] == "0.2.0"
        assert meta["seed"] == 42
        assert "python_version" in meta
        assert "platform" in meta
        assert "pip_freeze_hash" in meta
        assert meta["schema_version"] == 2
        assert meta["config_yaml"] == _MINIMAL_YAML

    def test_meta_json_stored_in_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "live.db"
        meta = {"app_version": "0.2.0", "seed": 99}
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="meta_run",
            params_yaml=_MINIMAL_YAML,
            results_root=tmp_path,
            T=1,
            meta_json=json.dumps(meta),
        )
        sink.close()

        # Read back meta_json from DB
        conn = _connect_sqlite(db_path)
        try:
            row = conn.execute(
                "SELECT meta_json FROM run_meta WHERE run_id='meta_run'"
            ).fetchone()
            assert row is not None
            stored = json.loads(row[0])
            assert stored["seed"] == 99
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 5. Status transition tests
# ---------------------------------------------------------------------------
class TestStatusTransitions:
    """Verify run status field behavior."""

    def test_stop_reason_persisted(self, tmp_path: Path) -> None:
        db_path = tmp_path / "live.db"
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="stop_run",
            params_yaml=_MINIMAL_YAML,
            results_root=tmp_path,
            T=1,
        )
        sink.set_status(state="stopped", message="user cancelled", stop_reason="user_stop")
        sink.close()

        status = read_status(db_path)
        assert status is not None
        assert status.state == "stopped"
        assert status.stop_reason == "user_stop"
        assert status.stopped_at is not None

    def test_error_state_has_stopped_at(self, tmp_path: Path) -> None:
        db_path = tmp_path / "live.db"
        sink = SQLiteLiveSink(
            db_path=db_path,
            run_id="err_run",
            params_yaml=_MINIMAL_YAML,
            results_root=tmp_path,
            T=1,
        )
        sink.set_status(state="error", message="boom", stop_reason="exception")
        sink.close()

        status = read_status(db_path)
        assert status is not None
        assert status.state == "error"
        assert status.stopped_at is not None
        assert status.stop_reason == "exception"

    def test_worker_flushes_final_snapshot_when_live_every_skips_terminal_block(self, tmp_path: Path) -> None:
        from abm_webapp.worker import run_simulation_process

        base_config_path = Path("abm_results/scenarios/test.yml")
        config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
        assert isinstance(config, dict)
        simulate_block = dict(config["simulate"])
        simulate_block.update(
            {
                "T": 6,
                "skip_step": 0,
                "visualize": False,
                "verbose": False,
                "N_LP": 10,
                "p_jit": 0.0,
                "initial_binom_N": 20,
                "initial_total_L": 10_000.0,
            }
        )
        config["simulate"] = simulate_block
        config_yaml = yaml.safe_dump(config, sort_keys=False)

        run_root = tmp_path / "web_runs" / "terminal_snapshot"
        run_simulation_process(
            run_id="terminal_snapshot",
            run_root=str(run_root),
            config_yaml=config_yaml,
            stop_event=Event(),
            live_every=4,
            log_flush_every=50,
        )

        status = read_status(run_root / "live.db")
        rows = read_metrics(run_root / "live.db")
        assert status is not None
        assert status.state == "finished"
        assert status.t_last == 5
        assert [row["t"] for row in rows] == [0, 4, 5]


# ---------------------------------------------------------------------------
# 6. Migration safety test
# ---------------------------------------------------------------------------
class TestMigration:
    """Verify that old DBs (missing columns) are migrated safely."""

    def test_v1_db_migrated(self, tmp_path: Path) -> None:
        """Simulate a V1 DB without new columns and verify migration."""
        db_path = tmp_path / "live.db"
        conn = _connect_sqlite(db_path)
        # Create minimal V1-like schema (no pid, heartbeat, etc.)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                params_yaml TEXT NOT NULL,
                results_root TEXT NOT NULL,
                T INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_status (
                run_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                t_last INTEGER NOT NULL,
                message TEXT,
                updated_at TEXT NOT NULL,
                log_path TEXT
            );
            CREATE TABLE IF NOT EXISTS metrics (
                t INTEGER PRIMARY KEY,
                dex_price REAL,
                cex_price REAL,
                band_lo REAL,
                band_hi REAL,
                sr_pnl_step REAL,
                noise_pnl_step REAL,
                arb_pnl_step REAL,
                lp_pnl_active REAL,
                lp_pnl_passive REAL,
                jiter_pnl REAL,
                fee REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO run_meta VALUES (?, ?, ?, ?, ?)",
            ("old_run", _utc_now_iso(), _MINIMAL_YAML, str(tmp_path), 100),
        )
        conn.execute(
            "INSERT INTO run_status VALUES (?, ?, ?, ?, ?, ?)",
            ("old_run", "running", 50, "", _utc_now_iso(), ""),
        )
        conn.commit()
        conn.close()

        # Now apply migration via _ensure_schema
        conn2 = _connect_sqlite(db_path)
        _ensure_schema(conn2)
        conn2.close()

        # Verify new columns exist and status can be read with them
        status = read_status(db_path)
        assert status is not None
        assert status.run_id == "old_run"
        assert status.pid is None  # was not set in V1
        assert status.heartbeat_at is None

        # Verify schema version was set
        assert read_schema_version(db_path) == SCHEMA_VERSION
