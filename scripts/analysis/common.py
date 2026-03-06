"""scripts/analysis/common.py — Shared helpers for the analysis package.

Provides:
- ``run_multi_seed``: parallel multi-seed simulation runner.
- Series manipulation (``slice_series``, ``diff_cumulative``).
- Plotly figure persistence (``save_figure``).
- Consistent styling constants.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Keys that analysis modules may need from the simulate() return dict.
# Workers in ``run_multi_seed`` extract (at least) these.
# ---------------------------------------------------------------------------
ANALYSIS_KEYS: List[str] = [
    # Hedged PnL (cumulative)
    "lp_pnl_total", "lp_pnl_active", "lp_pnl_passive",
    # Unhedged PnL (cumulative)
    "lp_unhedged_total", "lp_unhedged_active", "lp_unhedged_passive",
    # Cumulative fees (token-1 valued)
    "lp_fee_value_total_series", "lp_fee_value_active_series", "lp_fee_value_passive_series",
    # Cumulative LVR
    "lp_lvr_total_series", "lp_lvr_active_series", "lp_lvr_passive_series",
    # CEX instantaneous volatility (per-block)
    "cex_sigma_series",
    # Fee level applied (per-block)
    "fee_series", "fee_mode",
    # JIT
    "jiter_pnl_series", "jiter_fee_value_series",
    "jiter_flash_fee_paid_series", "jiter_activity_cum",
    # Other agents (cumulative PnL)
    "arb_pnl_cum", "smart_router_pnl_cum", "noise_trader_pnl_cum",
    # DEX share
    "smart_router_dex_share_series", "smart_router_dex_share_steps",
    "smart_router_dex_share_mean", "smart_router_dex_share_overall",
    # Scalar counts
    "total_arb_swaps", "arb_no_op_in_band", "arb_swaps_rejected_profitability",
    "total_noise_trader_swaps", "total_smart_router_swaps", "total_jit_trades_executed",
    # Prices
    "DEX_price", "CEX_price",
]

PAPER_SCALAR_RECORD_KEYS: List[str] = [
    "lp_pnl_active_final",
    "lp_pnl_passive_final",
    "jiter_pnl_final",
    "noise_trader_pnl_cum_final",
    "smart_router_pnl_cum_final",
    "lp_fee_value_active_final",
    "lp_fee_value_passive_final",
    "lp_fee_value_total_final",
    "fee_mean",
    "smart_router_dex_share_mean",
]

PAPER_VOLATILITY_RECORD_KEYS: List[str] = [
    "cex_sigma_series",
    "lp_fee_value_active_series",
    "lp_fee_value_passive_series",
    "lp_lvr_active_series",
    "lp_lvr_passive_series",
]

PAPER_RECORD_KEYS: List[str] = PAPER_SCALAR_RECORD_KEYS + PAPER_VOLATILITY_RECORD_KEYS

# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_white"
FONT = dict(size=16, color="black")
LEGEND_STYLE = dict(
    orientation="h", yanchor="bottom", y=1.02, x=0,
    font=dict(size=14, color="black"),
)
GRID_STYLE = dict(showgrid=True, gridcolor="#e1e1e1", gridwidth=1)

# Agent color palette (matches run_multiple.py conventions where possible)
COLORS = {
    "passive_lp": "#8c564b",
    "active_lp": "#9467bd",
    "jiter": "#d62728",
    "arb": "#2ca02c",
    "smart_router": "#1f77b4",
    "noise_trader": "#ff7f0e",
    "fees": "#17becf",
    "lvr": "#e377c2",
}


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------

def slice_series(values: Sequence[float], skip: int) -> np.ndarray:
    """Return *values* with the first *skip* elements removed."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    s0 = max(0, min(int(skip), int(arr.size)))
    return arr[s0:]


def diff_cumulative(series: Sequence[float]) -> np.ndarray:
    """Convert a cumulative series to per-step increments (Δ[0] = series[0])."""
    arr = np.asarray(series, dtype=float)
    if arr.size <= 1:
        return arr.copy()
    out = np.empty_like(arr)
    out[0] = arr[0]
    out[1:] = np.diff(arr)
    return out


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    s = hex_color.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Figure persistence
# ---------------------------------------------------------------------------

def save_figure(
    fig,
    out_dir: Path,
    name: str,
    *,
    width: int = 1400,
    height: int = 900,
    scale: float = 1.0,
) -> None:
    """Save a Plotly figure as HTML + PNG under *out_dir*."""
    out_dir = Path(out_dir)
    (out_dir / "html").mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(**GRID_STYLE)
    fig.write_html(str(out_dir / "html" / f"{name}.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(str(out_dir / "png" / f"{name}.png"),
                        width=width, height=height, scale=scale)
    except Exception:
        pass  # kaleido may not be installed


def apply_default_layout(fig, *, title: str = "", xaxis_title: str = "Block",
                         yaxis_title: str = "Token-1 value") -> None:
    """Apply consistent styling to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=LEGEND_STYLE,
        font=FONT,
    )


# ---------------------------------------------------------------------------
# Multi-seed parallel runner
# ---------------------------------------------------------------------------

def _silent_tqdm(iterable=None, **kwargs):
    if iterable is None:
        return range(int(kwargs.get("total", 0)))
    return iterable


def _float_or_nan(value: Any) -> float:
    """Convert *value* to float, falling back to NaN for missing data."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _paper_scalar_payload(seed: int, out: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one simulation output dict to the scalar keys used by paper plots."""
    return {
        "seed": int(seed),
        "lp_pnl_active": _float_or_nan(out.get("lp_pnl_active_final")),
        "lp_pnl_passive": _float_or_nan(out.get("lp_pnl_passive_final")),
        "jiter_pnl_series": _float_or_nan(out.get("jiter_pnl_final")),
        "noise_trader_pnl_cum": _float_or_nan(out.get("noise_trader_pnl_cum_final")),
        "smart_router_pnl_cum": _float_or_nan(out.get("smart_router_pnl_cum_final")),
        "lp_fee_value_active_series": _float_or_nan(out.get("lp_fee_value_active_final")),
        "lp_fee_value_passive_series": _float_or_nan(out.get("lp_fee_value_passive_final")),
        "lp_fee_value_total_series": _float_or_nan(out.get("lp_fee_value_total_final")),
        "fee_series": _float_or_nan(out.get("fee_mean")),
        "smart_router_dex_share_mean": _float_or_nan(out.get("smart_router_dex_share_mean")),
    }


def _write_volatility_artifact(out: Dict[str, Any], artifact_path: Path, skip: int) -> None:
    """Persist the exact per-block inputs needed for volatility-conditioned plots."""
    sigma = slice_series(out.get("cex_sigma_series", []), skip)
    fee_passive = slice_series(out.get("lp_fee_value_passive_series", []), skip)
    lvr_passive = slice_series(out.get("lp_lvr_passive_series", []), skip)
    fee_active = slice_series(out.get("lp_fee_value_active_series", []), skip)
    lvr_active = slice_series(out.get("lp_lvr_active_series", []), skip)

    n_passive = min(sigma.size, fee_passive.size, lvr_passive.size)
    n_active = min(sigma.size, fee_active.size, lvr_active.size)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        sigma_passive=sigma[:n_passive],
        d_fee_passive=diff_cumulative(fee_passive[:n_passive]),
        d_lvr_passive=diff_cumulative(lvr_passive[:n_passive]),
        sigma_active=sigma[:n_active],
        d_fee_active=diff_cumulative(fee_active[:n_active]),
        d_lvr_active=diff_cumulative(lvr_active[:n_active]),
    )


def _analysis_worker(
    seed: int,
    *,
    base_params: Dict[str, Any],
    tmp_root: Path,
    keys: List[str],
    mode: str = "extract",
    artifact_dir: Optional[Path] = None,
    skip: int = 0,
) -> Dict[str, Any]:
    """Worker that runs one seed and extracts *keys* from the output."""
    from scripts import run as _run_module
    _run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]

    params = dict(base_params)
    params["seed"] = int(seed)
    run_dir = tmp_root / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params["results_root"] = run_dir
    if mode == "paper":
        params["record_keys"] = list(PAPER_RECORD_KEYS)

    out = _run_module.simulate(**params)

    payload: Dict[str, Any] = {"seed": int(seed)}
    if mode == "paper":
        if artifact_dir is None:
            raise ValueError("artifact_dir is required when mode='paper'.")
        artifact_path = Path(artifact_dir) / f"seed_{int(seed)}.npz"
        _write_volatility_artifact(out, artifact_path=artifact_path, skip=skip)
        payload["scalars"] = _paper_scalar_payload(int(seed), out)
        payload["artifact_path"] = str(artifact_path)
    else:
        for key in keys:
            val = out.get(key)
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                payload[key] = val.copy()
            elif isinstance(val, list):
                payload[key] = list(val)
            else:
                payload[key] = val

    shutil.rmtree(run_dir, ignore_errors=True)
    return payload


def iter_multi_seed(
    base_params: Dict[str, Any],
    n_seeds: int,
    *,
    seed_base: int = 1,
    seed_step: int = 1,
    max_workers: int = 4,
    keys: Optional[List[str]] = None,
    tmp_root: Optional[Path] = None,
    mode: str = "extract",
    artifact_dir: Optional[Path] = None,
    skip: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Yield multi-seed simulation outputs as workers finish.

    Parameters
    ----------
    base_params : dict
        Base keyword arguments passed to ``simulate()``. The helper overrides
        ``seed``, ``results_root``, and disables visualization side effects.
    n_seeds : int
        Number of deterministic seeds to execute.
    seed_base : int
        First seed in the arithmetic progression.
    seed_step : int
        Step size between consecutive seeds.
    max_workers : int
        Maximum number of worker processes used by ``ProcessPoolExecutor``.
    keys : list[str] or None
        Output keys to extract in the default ``"extract"`` mode.
    tmp_root : Path or None
        Temporary root for per-seed scratch folders.
    mode : str
        ``"extract"`` returns the requested keys directly. ``"paper"`` reduces
        scalar outputs and writes volatility intermediates to ``artifact_dir``.
    artifact_dir : Path or None
        Destination directory for per-seed volatility artifacts in ``"paper"``
        mode.
    skip : int
        Burn-in blocks dropped before writing volatility artifacts.

    Returns
    -------
    Iterator[dict]
        One payload per completed seed. Payload order follows completion order.

    Notes
    -----
    - Work units are independent seeds, so execution stays reproducible.
    - The generator must be fully consumed (or closed) to guarantee cleanup of
      the temporary per-seed directories.

    Examples
    --------
    >>> params = {"config_name": "demo", "seed": 1}
    >>> iterator = iter_multi_seed(params, 0)
    >>> list(iterator)
    []
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if keys is None:
        keys = list(ANALYSIS_KEYS)
    if mode not in {"extract", "paper"}:
        raise ValueError(f"Unsupported iter_multi_seed mode: {mode!r}")
    if mode == "paper" and artifact_dir is None:
        raise ValueError("artifact_dir is required when mode='paper'.")

    params = dict(base_params)
    params["light_mode"] = False
    params["visualize"] = False
    params["verbose"] = False
    params["liquidity_for_gif"] = False

    if tmp_root is None:
        tmp_root = Path(f"/tmp/abm_analysis_{os.getpid()}")
    tmp_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed_base + i * seed_step) for i in range(n_seeds)]

    results: List[Dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _analysis_worker,
                    seed,
                    base_params=params,
                    tmp_root=tmp_root,
                    keys=keys,
                    mode=mode,
                    artifact_dir=artifact_dir,
                    skip=skip,
                ): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                yield fut.result()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def run_multi_seed(
    base_params: Dict[str, Any],
    n_seeds: int,
    *,
    seed_base: int = 1,
    seed_step: int = 1,
    max_workers: int = 4,
    keys: Optional[List[str]] = None,
    tmp_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run ``simulate()`` across seeds and materialize the extracted outputs.

    Parameters
    ----------
    base_params : dict
        Base keyword arguments passed to ``simulate()`` for each seed.
    n_seeds : int
        Number of independent seeds.
    seed_base : int
        First seed in the arithmetic progression.
    seed_step : int
        Step size between consecutive seeds.
    max_workers : int
        Maximum number of worker processes used by ``ProcessPoolExecutor``.
    keys : list[str] or None
        Output keys to extract. ``None`` defaults to ``ANALYSIS_KEYS``.
    tmp_root : Path or None
        Temporary root for per-seed scratch folders.

    Returns
    -------
    list[dict]
        One extracted payload per seed, sorted by seed.

    Notes
    -----
    - This is the historical materializing API kept for compatibility.
    - For memory-sensitive workflows, prefer ``iter_multi_seed`` or the paper
      figure runner's streaming path.

    Examples
    --------
    >>> run_multi_seed({}, 0)
    []
    """
    results = list(
        iter_multi_seed(
            base_params,
            n_seeds,
            seed_base=seed_base,
            seed_step=seed_step,
            max_workers=max_workers,
            keys=keys,
            tmp_root=tmp_root,
        )
    )
    results.sort(key=lambda r: int(r.get("seed", 0)))
    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def align_multi_run_series(
    results: List[Dict[str, Any]],
    key: str,
    skip: int = 0,
) -> np.ndarray:
    """Stack a series across runs into a (n_runs, min_len) array after skip."""
    sliced = [slice_series(r.get(key, []), skip) for r in results
              if len(r.get(key, [])) > 0]
    if not sliced:
        return np.empty((0, 0), dtype=float)
    min_len = min(s.size for s in sliced)
    if min_len == 0:
        return np.empty((len(sliced), 0), dtype=float)
    return np.stack([s[:min_len] for s in sliced], axis=0)


def final_values(
    results: List[Dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Extract the last element of a series across runs."""
    vals = []
    for r in results:
        s = r.get(key, [])
        if isinstance(s, (list, np.ndarray)) and len(s) > 0:
            vals.append(float(s[-1]))
        elif isinstance(s, (int, float)):
            vals.append(float(s))
    return np.array(vals, dtype=float)
