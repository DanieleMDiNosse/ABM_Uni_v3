from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _ensure_schema(conn: sqlite3.Connection) -> None:
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
            log_path TEXT,
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
    # Migrations for older DBs (created before new columns existed).
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
        try:
            conn.execute(f"ALTER TABLE metrics ADD COLUMN {col_def};")
        except sqlite3.OperationalError as exc:
            # SQLite does not support IF NOT EXISTS for ADD COLUMN in older versions.
            msg = str(exc).lower()
            if "duplicate column name" in msg:
                pass
            else:
                raise
    conn.commit()


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    state: str
    t_last: int
    message: str
    updated_at: str
    log_path: str


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
    ) -> None:
        self._db_path = Path(db_path)
        self._run_id = str(run_id)
        self._commit_every = max(1, int(commit_every))
        self._conn = _connect_sqlite(self._db_path)
        _ensure_schema(self._conn)

        self._pending_metrics: List[Tuple[Any, ...]] = []
        self._t_last: int = -1

        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_meta(run_id, created_at, params_yaml, results_root, T)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self._run_id, _utc_now_iso(), str(params_yaml), str(results_root), int(T)),
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_status(run_id, state, t_last, message, updated_at, log_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self._run_id, "running", -1, "", _utc_now_iso(), ""),
        )
        self._conn.commit()

    @property
    def run_id(self) -> str:
        return self._run_id

    def set_log_path(self, log_path: str) -> None:
        """Persist the log file path so the UI can tail it."""
        self._conn.execute(
            """
            UPDATE run_status
            SET log_path = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (str(log_path), _utc_now_iso(), self._run_id),
        )
        self._conn.commit()

    def record_step(self, row: Dict[str, Any]) -> None:
        """
        Record one simulation step row.

        Parameters
        ----------
        row
            Mapping with at least:
            - t (int)
            - dex_price, cex_price, band_lo, band_hi (float)
            - sr_pnl_step, noise_pnl_step, arb_pnl_step (float)
            - lp_pnl_active, lp_pnl_passive, jiter_pnl (float)
            - fee (float)
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

    def set_status(self, *, state: str, message: str = "") -> None:
        """Update run status (e.g., running/finished/error/stopped)."""
        self._conn.execute(
            """
            UPDATE run_status
            SET state = ?, t_last = ?, message = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (str(state), int(self._t_last), str(message), _utc_now_iso(), self._run_id),
        )
        self._conn.commit()

    def flush(self) -> None:
        """Flush buffered metric rows and update t_last."""
        if self._pending_metrics:
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
            self._pending_metrics.clear()
        self._conn.execute(
            """
            UPDATE run_status
            SET t_last = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (int(self._t_last), _utc_now_iso(), self._run_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Flush and close the underlying SQLite connection."""
        self.flush()
        self._conn.close()


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
        row = conn.execute(
            """
            SELECT run_id, state, t_last, COALESCE(message,''), updated_at, COALESCE(log_path,'')
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
        Path to `live.db`.
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
            # Apply the cap in SQL for the common "load latest window" path.
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
            rows = rows[-int(limit) :]
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
