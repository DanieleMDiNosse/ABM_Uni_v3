"""
SQLite-backed storage for the ABM webapp.

Responsibilities:
- Per-run ``live.db`` (metrics, status, metadata) used by both the worker
  process (writer) and the Dash UI process (reader).
- WAL mode for concurrent read/write without blocking.
- Explicit schema versioning with safe migrations.
- Heartbeat + PID tracking for crash recovery.
- Retry-on-locked writes.

This module has **no** dependency on Dash or Flask.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION: int = 2  # bump when DDL changes
_RETRY_ATTEMPTS: int = 5
_RETRY_DELAY_S: float = 0.15
HEARTBEAT_STALE_SECONDS: float = 60.0  # worker must update more often than this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL enabled (better for concurrent reads).

    Notes
    -----
    - The simulation process is the single writer.
    - The Dash process reads concurrently, so WAL avoids reader/writer blocking.
    """
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _retry_execute(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute a single statement with retry on ``database is locked``."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
            else:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


def _retry_commit(conn: sqlite3.Connection) -> None:
    """Commit with retry on ``database is locked``."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_DELAY_S * (attempt + 1))
            else:
                raise


# ---------------------------------------------------------------------------
# Schema creation + migration
# ---------------------------------------------------------------------------
def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_meta (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            params_yaml TEXT NOT NULL,
            results_root TEXT NOT NULL,
            T INTEGER NOT NULL,
            meta_json TEXT
        );

        CREATE TABLE IF NOT EXISTS run_status (
            run_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            t_last INTEGER NOT NULL,
            message TEXT,
            updated_at TEXT NOT NULL,
            log_path TEXT,
            pid INTEGER,
            heartbeat_at TEXT,
            stop_reason TEXT,
            stopped_at TEXT,
            FOREIGN KEY(run_id) REFERENCES run_meta(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS metrics (
            t INTEGER PRIMARY KEY,
            dex_price REAL,
            cex_price REAL,
            cex_sigma REAL,
            band_lo REAL,
            band_hi REAL,
            sr_pnl_step REAL,
            noise_pnl_step REAL,
            arb_pnl_step REAL,
            lp_pnl_active REAL,
            lp_pnl_passive REAL,
            lp_unhedged_active REAL,
            lp_unhedged_passive REAL,
            lp_fee_value_total REAL,
            lp_lvr_total REAL,
            jiter_pnl REAL,
            dex_notional_y REAL,
            d_lvr_total REAL,
            d_fee_value_total REAL,
            trader_exec_count INTEGER,
            arb_exec_count INTEGER,
            sr_exec_count INTEGER,
            noise_exec_count INTEGER,
            sr_cex_exec_count INTEGER,
            sr_dex_exec_count INTEGER,
            fee REAL,
            fee_sigma REAL,
            fee_basis_ticks REAL,
            fee_signal REAL
        );
        """
    )

    # ── Column migrations (for DBs created before V2 columns existed) ──
    _migrate_columns(conn)

    # ── Record schema version ──
    conn.execute(
        "INSERT OR REPLACE INTO schema_info(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add columns that may be missing from older DBs."""
    # Metrics columns
    for col_def in (
        "cex_sigma REAL",
        "lp_unhedged_active REAL",
        "lp_unhedged_passive REAL",
        "lp_fee_value_total REAL",
        "lp_lvr_total REAL",
        "dex_notional_y REAL",
        "d_lvr_total REAL",
        "d_fee_value_total REAL",
        "trader_exec_count INTEGER",
        "arb_exec_count INTEGER",
        "sr_exec_count INTEGER",
        "noise_exec_count INTEGER",
        "sr_cex_exec_count INTEGER",
        "sr_dex_exec_count INTEGER",
        "fee_sigma REAL",
        "fee_basis_ticks REAL",
        "fee_signal REAL",
    ):
        _try_add_column(conn, "metrics", col_def)

    # run_meta columns
    _try_add_column(conn, "run_meta", "meta_json TEXT")

    # run_status columns
    for col_def in (
        "pid INTEGER",
        "heartbeat_at TEXT",
        "stop_reason TEXT",
        "stopped_at TEXT",
    ):
        _try_add_column(conn, "run_status", col_def)


def _try_add_column(conn: sqlite3.Connection, table: str, col_def: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def};")
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if "duplicate column name" in msg:
            pass
        else:
            raise


def read_schema_version(db_path: Path) -> Optional[int]:
    """Read schema_version from an existing DB, or None if unavailable."""
    if not db_path.exists():
        return None
    try:
        conn = _connect_sqlite(db_path)
        try:
            row = conn.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
            return int(row[0]) if row else None
        finally:
            conn.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunStatus:
    run_id: str
    state: str
    t_last: int
    message: str
    updated_at: str
    log_path: str
    pid: Optional[int] = None
    heartbeat_at: Optional[str] = None
    stop_reason: Optional[str] = None
    stopped_at: Optional[str] = None


# Valid state transitions
TERMINAL_STATES = frozenset({"finished", "stopped", "error", "abandoned"})
VALID_STATES = frozenset({"running", "stopping", "finished", "stopped", "error", "abandoned"})


# ---------------------------------------------------------------------------
# Writer (used inside the worker process)
# ---------------------------------------------------------------------------
class SQLiteLiveSink:
    """
    SQLite-backed live sink for streaming simulation outputs to a Dash UI.

    Parameters
    ----------
    db_path
        Path to the SQLite database file to create/update.
    run_id
        Unique run identifier (used as primary key).
    params_yaml
        Full YAML configuration used for the run (stored for reproducibility).
    results_root
        Output directory for the run (logs, db, optional artifacts).
    T
        Requested number of simulation blocks.
    commit_every
        Flush buffered inserts every N rows.
    meta_json
        Optional JSON string with extended run metadata.

    Notes
    -----
    - This object is designed to be created and used inside the *simulation process*.
    - Reads should be performed by opening separate connections in the UI process.
    """

    def __init__(
        self,
        *,
        db_path: Path,
        run_id: str,
        params_yaml: str,
        results_root: Path,
        T: int,
        commit_every: int = 50,
        meta_json: Optional[str] = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._run_id = str(run_id)
        self._commit_every = max(1, int(commit_every))
        self._conn = _connect_sqlite(self._db_path)
        _ensure_schema(self._conn)

        self._pending_metrics: List[Tuple[Any, ...]] = []
        self._t_last: int = -1

        _retry_execute(
            self._conn,
            """
            INSERT OR REPLACE INTO run_meta(run_id, created_at, params_yaml, results_root, T, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self._run_id, _utc_now_iso(), str(params_yaml), str(results_root), int(T), meta_json),
        )
        _retry_execute(
            self._conn,
            """
            INSERT OR REPLACE INTO run_status(run_id, state, t_last, message, updated_at, log_path, pid, heartbeat_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self._run_id, "running", -1, "", _utc_now_iso(), "", os.getpid(), _utc_now_iso()),
        )
        _retry_commit(self._conn)

    @property
    def run_id(self) -> str:
        return self._run_id

    def set_log_path(self, log_path: str) -> None:
        """Persist the log file path so the UI can tail it."""
        _retry_execute(
            self._conn,
            """
            UPDATE run_status
            SET log_path = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (str(log_path), _utc_now_iso(), self._run_id),
        )
        _retry_commit(self._conn)

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp (call periodically from worker)."""
        _retry_execute(
            self._conn,
            "UPDATE run_status SET heartbeat_at = ? WHERE run_id = ?",
            (_utc_now_iso(), self._run_id),
        )
        _retry_commit(self._conn)

    def record_step(self, row: Dict[str, Any]) -> None:
        """
        Record one simulation step row.

        Parameters
        ----------
        row
            Mapping with at least ``t`` (int) and other metric columns.
        """
        t = int(row["t"])
        self._t_last = max(self._t_last, t)
        self._pending_metrics.append(
            (
                t,
                float(row.get("dex_price")) if row.get("dex_price") is not None else None,
                float(row.get("cex_price")) if row.get("cex_price") is not None else None,
                float(row.get("cex_sigma")) if row.get("cex_sigma") is not None else None,
                float(row.get("band_lo")) if row.get("band_lo") is not None else None,
                float(row.get("band_hi")) if row.get("band_hi") is not None else None,
                float(row.get("sr_pnl_step")) if row.get("sr_pnl_step") is not None else None,
                float(row.get("noise_pnl_step")) if row.get("noise_pnl_step") is not None else None,
                float(row.get("arb_pnl_step")) if row.get("arb_pnl_step") is not None else None,
                float(row.get("lp_pnl_active")) if row.get("lp_pnl_active") is not None else None,
                float(row.get("lp_pnl_passive")) if row.get("lp_pnl_passive") is not None else None,
                float(row.get("lp_unhedged_active")) if row.get("lp_unhedged_active") is not None else None,
                float(row.get("lp_unhedged_passive")) if row.get("lp_unhedged_passive") is not None else None,
                float(row.get("lp_fee_value_total")) if row.get("lp_fee_value_total") is not None else None,
                float(row.get("lp_lvr_total")) if row.get("lp_lvr_total") is not None else None,
                float(row.get("jiter_pnl")) if row.get("jiter_pnl") is not None else None,
                float(row.get("dex_notional_y")) if row.get("dex_notional_y") is not None else None,
                float(row.get("d_lvr_total")) if row.get("d_lvr_total") is not None else None,
                float(row.get("d_fee_value_total")) if row.get("d_fee_value_total") is not None else None,
                int(row.get("trader_exec_count")) if row.get("trader_exec_count") is not None else None,
                int(row.get("arb_exec_count")) if row.get("arb_exec_count") is not None else None,
                int(row.get("sr_exec_count")) if row.get("sr_exec_count") is not None else None,
                int(row.get("noise_exec_count")) if row.get("noise_exec_count") is not None else None,
                int(row.get("sr_cex_exec_count")) if row.get("sr_cex_exec_count") is not None else None,
                int(row.get("sr_dex_exec_count")) if row.get("sr_dex_exec_count") is not None else None,
                float(row.get("fee")) if row.get("fee") is not None else None,
                float(row.get("fee_sigma")) if row.get("fee_sigma") is not None else None,
                float(row.get("fee_basis_ticks")) if row.get("fee_basis_ticks") is not None else None,
                float(row.get("fee_signal")) if row.get("fee_signal") is not None else None,
            )
        )

        if len(self._pending_metrics) >= self._commit_every:
            self.flush()

    def set_status(
        self,
        *,
        state: str,
        message: str = "",
        stop_reason: Optional[str] = None,
    ) -> None:
        """Update run status (e.g., running/stopping/finished/error/stopped/abandoned)."""
        now = _utc_now_iso()
        stopped_at = now if state in TERMINAL_STATES else None
        _retry_execute(
            self._conn,
            """
            UPDATE run_status
            SET state = ?, t_last = ?, message = ?, updated_at = ?,
                heartbeat_at = ?, stop_reason = ?, stopped_at = ?
            WHERE run_id = ?
            """,
            (
                str(state), int(self._t_last), str(message), now,
                now, stop_reason, stopped_at, self._run_id,
            ),
        )
        _retry_commit(self._conn)

    def flush(self) -> None:
        """Flush buffered metric rows and update t_last."""
        if self._pending_metrics:
            for attempt in range(_RETRY_ATTEMPTS):
                try:
                    self._conn.executemany(
                        """
                        INSERT OR REPLACE INTO metrics(
                            t, dex_price, cex_price, cex_sigma, band_lo, band_hi,
                            sr_pnl_step, noise_pnl_step, arb_pnl_step,
                            lp_pnl_active, lp_pnl_passive,
                            lp_unhedged_active, lp_unhedged_passive,
                            lp_fee_value_total, lp_lvr_total, jiter_pnl,
                            dex_notional_y, d_lvr_total, d_fee_value_total,
                            trader_exec_count, arb_exec_count, sr_exec_count,
                            noise_exec_count, sr_cex_exec_count, sr_dex_exec_count,
                            fee,
                            fee_sigma, fee_basis_ticks, fee_signal
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        self._pending_metrics,
                    )
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt < _RETRY_ATTEMPTS - 1:
                        time.sleep(_RETRY_DELAY_S * (attempt + 1))
                    else:
                        raise
            self._pending_metrics.clear()

        _retry_execute(
            self._conn,
            """
            UPDATE run_status
            SET t_last = ?, updated_at = ?, heartbeat_at = ?
            WHERE run_id = ?
            """,
            (int(self._t_last), _utc_now_iso(), _utc_now_iso(), self._run_id),
        )
        _retry_commit(self._conn)

    def close(self) -> None:
        """Flush and close the underlying SQLite connection."""
        self.flush()
        self._conn.close()


# ---------------------------------------------------------------------------
# Readers (used by the Dash UI process)
# ---------------------------------------------------------------------------
def read_status(db_path: Path) -> Optional[RunStatus]:
    """
    Read current run status from the SQLite DB.

    Returns
    -------
    RunStatus or None if the DB does not exist / is empty.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    conn = _connect_sqlite(db_path)
    try:
        # Check which columns exist
        cols = {r[1] for r in conn.execute("PRAGMA table_info(run_status);").fetchall()}
        has_pid = "pid" in cols
        has_heartbeat = "heartbeat_at" in cols
        has_stop_reason = "stop_reason" in cols
        has_stopped_at = "stopped_at" in cols

        pid_sel = "pid" if has_pid else "NULL AS pid"
        hb_sel = "heartbeat_at" if has_heartbeat else "NULL AS heartbeat_at"
        sr_sel = "stop_reason" if has_stop_reason else "NULL AS stop_reason"
        sa_sel = "stopped_at" if has_stopped_at else "NULL AS stopped_at"

        row = conn.execute(
            f"""
            SELECT run_id, state, t_last, COALESCE(message,''), updated_at,
                   COALESCE(log_path,''), {pid_sel}, {hb_sel}, {sr_sel}, {sa_sel}
            FROM run_status
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return RunStatus(
            run_id=str(row[0]),
            state=str(row[1]),
            t_last=int(row[2]),
            message=str(row[3]),
            updated_at=str(row[4]),
            log_path=str(row[5]),
            pid=int(row[6]) if row[6] is not None else None,
            heartbeat_at=str(row[7]) if row[7] is not None else None,
            stop_reason=str(row[8]) if row[8] is not None else None,
            stopped_at=str(row[9]) if row[9] is not None else None,
        )
    finally:
        conn.close()


def read_metrics(
    db_path: Path,
    *,
    since_t: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Read metrics from the SQLite DB as a list of dicts.

    Parameters
    ----------
    db_path
        Path to ``live.db``.
    since_t
        If provided, only rows with t > since_t are returned.
    limit
        If provided, limits the number of returned rows (most recent).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = _connect_sqlite(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(metrics);").fetchall()}

        def _sel(name: str) -> str:
            return name if name in cols else f"NULL AS {name}"

        where = ""
        params: List[Any] = []
        if since_t is not None:
            where = "WHERE t > ?"
            params.append(int(since_t))

        base_sql = f"""
            SELECT
                t, dex_price, cex_price, {_sel("cex_sigma")}, band_lo, band_hi,
                sr_pnl_step, noise_pnl_step, arb_pnl_step,
                lp_pnl_active, lp_pnl_passive,
                {_sel("lp_unhedged_active")}, {_sel("lp_unhedged_passive")},
                {_sel("lp_fee_value_total")}, {_sel("lp_lvr_total")},
                jiter_pnl, {_sel("dex_notional_y")},
                {_sel("d_lvr_total")}, {_sel("d_fee_value_total")},
                {_sel("trader_exec_count")}, {_sel("arb_exec_count")},
                {_sel("sr_exec_count")}, {_sel("noise_exec_count")},
                {_sel("sr_cex_exec_count")}, {_sel("sr_dex_exec_count")},
                fee, {_sel("fee_sigma")}, {_sel("fee_basis_ticks")}, {_sel("fee_signal")}
            FROM metrics
            {where}
        """

        if limit is not None and since_t is None:
            sql = f"""
                SELECT *
                FROM (
                    {base_sql}
                    ORDER BY t DESC
                    LIMIT ?
                )
                ORDER BY t ASC
            """
            query_params: List[Any] = [int(limit)]
        else:
            sql = f"""
                {base_sql}
                ORDER BY t ASC
            """
            query_params = list(params)

        rows = conn.execute(sql, query_params).fetchall()
        if limit is not None and since_t is not None and len(rows) > int(limit):
            rows = rows[-int(limit):]
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                dict(
                    t=int(r[0]),
                    dex_price=r[1],
                    cex_price=r[2],
                    cex_sigma=r[3],
                    band_lo=r[4],
                    band_hi=r[5],
                    sr_pnl_step=r[6],
                    noise_pnl_step=r[7],
                    arb_pnl_step=r[8],
                    lp_pnl_active=r[9],
                    lp_pnl_passive=r[10],
                    lp_unhedged_active=r[11],
                    lp_unhedged_passive=r[12],
                    lp_fee_value_total=r[13],
                    lp_lvr_total=r[14],
                    jiter_pnl=r[15],
                    dex_notional_y=r[16],
                    d_lvr_total=r[17],
                    d_fee_value_total=r[18],
                    trader_exec_count=r[19],
                    arb_exec_count=r[20],
                    sr_exec_count=r[21],
                    noise_exec_count=r[22],
                    sr_cex_exec_count=r[23],
                    sr_dex_exec_count=r[24],
                    fee=r[25],
                    fee_sigma=r[26],
                    fee_basis_ticks=r[27],
                    fee_signal=r[28],
                )
            )
        return out
    finally:
        conn.close()


def tail_text_file(path: Path, *, max_bytes: int = 50_000) -> str:
    """
    Return the tail of a text file.

    Notes
    -----
    This uses a byte tail (not line-based) for robustness with large logs.
    """
    path = Path(path)
    if not path.exists():
        return ""
    max_bytes = max(1, int(max_bytes))
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")


# ---------------------------------------------------------------------------
# Crash recovery helpers (used by the webapp on startup)
# ---------------------------------------------------------------------------
def is_pid_alive(pid: Optional[int]) -> bool:
    """Check if a process with the given PID is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check
        return True
    except (OSError, PermissionError):
        return False


def is_heartbeat_stale(heartbeat_at: Optional[str], *, threshold_s: float = HEARTBEAT_STALE_SECONDS) -> bool:
    """Return True if the heartbeat timestamp is older than threshold."""
    if heartbeat_at is None:
        return True
    try:
        hb_dt = datetime.fromisoformat(heartbeat_at)
        if hb_dt.tzinfo is None:
            hb_dt = hb_dt.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - hb_dt).total_seconds()
        return elapsed > threshold_s
    except Exception:
        return True


def mark_abandoned(db_path: Path, *, message: str = "Detected stale/dead worker on startup.") -> bool:
    """
    Mark a run's status as ``abandoned`` in its DB file.

    Returns True if the status was actually changed.
    """
    if not db_path.exists():
        return False
    try:
        conn = _connect_sqlite(db_path)
        _ensure_schema(conn)
        try:
            now = _utc_now_iso()
            cur = conn.execute(
                """
                UPDATE run_status
                SET state = 'abandoned', message = ?, updated_at = ?,
                    stop_reason = 'crash_recovery', stopped_at = ?
                WHERE state IN ('running', 'stopping')
                """,
                (message, now, now),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    except Exception:
        return False


def scan_and_recover(web_runs_root: Path) -> List[str]:
    """
    Scan all run directories and mark any running/stopping runs with dead
    PIDs or stale heartbeats as ``abandoned``.

    Returns list of recovered run_ids.
    """
    recovered: List[str] = []
    if not web_runs_root.exists():
        return recovered

    for run_dir in sorted(web_runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        db_path = run_dir / "live.db"
        if not db_path.exists():
            continue
        status = read_status(db_path)
        if status is None:
            continue
        if status.state not in ("running", "stopping"):
            continue

        # Check if worker is actually dead
        pid_dead = not is_pid_alive(status.pid)
        hb_stale = is_heartbeat_stale(status.heartbeat_at)

        if pid_dead or hb_stale:
            reasons = []
            if pid_dead:
                reasons.append(f"PID {status.pid} is dead")
            if hb_stale:
                reasons.append(f"heartbeat stale ({status.heartbeat_at})")
            msg = f"Crash recovery: {'; '.join(reasons)}."
            if mark_abandoned(db_path, message=msg):
                recovered.append(status.run_id)

    return recovered
