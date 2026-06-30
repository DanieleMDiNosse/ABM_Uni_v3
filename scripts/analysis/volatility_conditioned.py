"""scripts/analysis/volatility_conditioned.py — Volatility-binned analysis.

Bins simulation blocks by the CEX instantaneous volatility σ_t and computes
the aggregate ΔLVR / ΔFees ratio within volatility quantile bins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import FONT, PLOTLY_TEMPLATE, diff_cumulative, slice_series

_COHORT_MAP = {
    "passive": ("lp_fee_value_passive_series", "lp_lvr_passive_series"),
    "active": ("lp_fee_value_active_series", "lp_lvr_active_series"),
    "total": ("lp_fee_value_total_series", "lp_lvr_total_series"),
}

_ARTIFACT_COHORT_MAP = {
    "passive": ("sigma_passive", "d_fee_passive", "d_lvr_passive"),
    "active": ("sigma_active", "d_fee_active", "d_lvr_active"),
}


def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5)
    return fig


def _cohort_arrays_from_results(
    results: Sequence[Dict[str, Any]],
    *,
    cohort: str,
    skip: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fee_key, lvr_key = _COHORT_MAP[cohort]
    arrays: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for result in results:
        sigma = slice_series(result.get("cex_sigma_series", []), skip)
        fees_cum = slice_series(result.get(fee_key, []), skip)
        lvr_cum = slice_series(result.get(lvr_key, []), skip)
        n_obs = min(sigma.size, fees_cum.size, lvr_cum.size)
        if n_obs < 2:
            continue
        arrays.append(
            (
                sigma[:n_obs],
                diff_cumulative(fees_cum[:n_obs]),
                diff_cumulative(lvr_cum[:n_obs]),
            )
        )
    return arrays


def _build_ratio_figure(
    *,
    cohort: str,
    n_bins: int,
    edges: np.ndarray,
    mean_ratio: Sequence[float],
    se_ratio: Sequence[float],
    bin_labels: Optional[Sequence[str]],
) -> go.Figure:
    labels = list(bin_labels) if bin_labels is not None else [f"Q{i+1}" for i in range(n_bins)]
    sigma_range_labels: List[str] = []
    for idx in range(n_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        sigma_range_labels.append(f"{labels[idx]}<br>[{lo:.2e}, {hi:.2e}]")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sigma_range_labels,
        y=list(mean_ratio),
        marker_color="#e377c2",
        error_y=dict(type="data", array=list(se_ratio), visible=True),
        name="ΔLVR/ΔFees",
    ))
    fig.add_hline(y=1, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Aggregate ΔLVR / ΔFees by σ_t quantile — {cohort} LPs ({n_bins} bins)",
        font=FONT,
        showlegend=False,
        height=450,
    )
    fig.update_xaxes(title_text="σ_t quantile")
    fig.update_yaxes(title_text="Ratio")
    return fig


def _stats_from_arrays(
    arrays: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    n_bins: int,
) -> Tuple[np.ndarray, List[float], List[float]]:
    sigma_cat = np.concatenate([item[0] for item in arrays])
    d_fee_cat = np.concatenate([item[1] for item in arrays])
    d_lvr_cat = np.concatenate([item[2] for item in arrays])

    edges = np.quantile(sigma_cat, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(sigma_cat, edges[1:-1], right=True)

    mean_ratio: List[float] = []
    se_ratio: List[float] = []
    for bin_id in range(n_bins):
        mask = bin_idx == bin_id
        if not np.any(mask):
            mean_ratio.append(float("nan"))
            se_ratio.append(0.0)
            continue
        fee_values = d_fee_cat[mask]
        lvr_values = d_lvr_cat[mask]
        total_fee = float(np.sum(fee_values))
        total_lvr = float(np.sum(lvr_values))
        ratio = total_lvr / total_fee if abs(total_fee) > 1e-12 else float("nan")
        mean_ratio.append(ratio)
        per_block_ratio = np.where(np.abs(fee_values) > 1e-12, lvr_values / fee_values, np.nan)
        valid = per_block_ratio[np.isfinite(per_block_ratio)]
        se_ratio.append(
            float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
        )
    return edges, mean_ratio, se_ratio


def volatility_binned_analysis(
    results: List[Dict[str, Any]],
    *,
    n_bins: int = 5,
    skip: int = 0,
    cohort: str = "total",
    bin_labels: Optional[List[str]] = None,
) -> go.Figure:
    """Build the in-memory volatility-binned ΔLVR/ΔFees bar chart.

    Parameters
    ----------
    results : list[dict]
        Per-seed simulation output dictionaries.
    n_bins : int
        Number of volatility quantile bins.
    skip : int
        Burn-in blocks discarded before constructing the bins.
    cohort : str
        LP cohort to analyze. Supported values are ``"passive"``, ``"active"``,
        and ``"total"``.
    bin_labels : list[str] or None
        Optional custom labels for the quantile bins.

    Returns
    -------
    go.Figure
        Bar chart with one bar per volatility quantile bin.

    Notes
    -----
    - The function preserves the historical analysis semantics exactly:
      it slices by ``skip`` first, differences cumulative fee/LVR paths second,
      then bins by quantiles of the pooled σ_t observations.
    - Standard errors are computed from valid per-block ratios within each bin.

    Examples
    --------
    >>> fig = volatility_binned_analysis([], cohort="passive")
    >>> isinstance(fig, go.Figure)
    True
    """
    arrays = _cohort_arrays_from_results(results, cohort=cohort, skip=skip)
    if not arrays:
        return _empty_figure()
    edges, mean_ratio, se_ratio = _stats_from_arrays(arrays, n_bins=n_bins)
    return _build_ratio_figure(
        cohort=cohort,
        n_bins=n_bins,
        edges=edges,
        mean_ratio=mean_ratio,
        se_ratio=se_ratio,
        bin_labels=bin_labels,
    )


def volatility_binned_analysis_from_artifacts(
    artifact_paths: Sequence[Path | str],
    *,
    n_bins: int = 5,
    cohort: str = "passive",
    bin_labels: Optional[List[str]] = None,
) -> go.Figure:
    """Build the volatility-binned chart from disk-backed per-seed artifacts.

    Parameters
    ----------
    artifact_paths : sequence[path-like]
        Paths to per-seed ``.npz`` artifacts created by the paper analysis
        worker. Each artifact stores the cohort-specific σ_t, ΔFees, and ΔLVR
        arrays after burn-in removal.
    n_bins : int
        Number of volatility quantile bins.
    cohort : str
        LP cohort to analyze. Supported values are ``"passive"`` and
        ``"active"``.
    bin_labels : list[str] or None
        Optional custom labels for the quantile bins.

    Returns
    -------
    go.Figure
        Bar chart with one bar per volatility quantile bin.

    Notes
    -----
    - The implementation uses two passes over the artifacts: one to compute the
      exact pooled quantile edges, and one to accumulate the per-bin sums and
      standard-error moments without materializing all fee/LVR blocks at once.
    - The result matches the in-memory analysis up to floating-point summation
      order.

    Examples
    --------
    >>> fig = volatility_binned_analysis_from_artifacts([], cohort="passive")
    >>> isinstance(fig, go.Figure)
    True
    """
    if cohort not in _ARTIFACT_COHORT_MAP:
        raise ValueError("Artifact-backed volatility analysis supports only 'passive' and 'active' cohorts.")
    sigma_key, fee_key, lvr_key = _ARTIFACT_COHORT_MAP[cohort]
    sigma_chunks: List[np.ndarray] = []
    normalized_paths = [Path(path) for path in artifact_paths]
    for artifact_path in normalized_paths:
        with np.load(artifact_path) as payload:
            sigma = np.asarray(payload[sigma_key], dtype=float)
            if sigma.size > 0:
                sigma_chunks.append(sigma)
    if not sigma_chunks:
        return _empty_figure()

    sigma_all = np.concatenate(sigma_chunks)
    edges = np.quantile(sigma_all, np.linspace(0, 1, n_bins + 1))
    fee_sum = np.zeros(n_bins, dtype=float)
    lvr_sum = np.zeros(n_bins, dtype=float)
    ratio_sum = np.zeros(n_bins, dtype=float)
    ratio_sumsq = np.zeros(n_bins, dtype=float)
    ratio_count = np.zeros(n_bins, dtype=int)

    for artifact_path in normalized_paths:
        with np.load(artifact_path) as payload:
            sigma = np.asarray(payload[sigma_key], dtype=float)
            d_fee = np.asarray(payload[fee_key], dtype=float)
            d_lvr = np.asarray(payload[lvr_key], dtype=float)
        n_obs = min(sigma.size, d_fee.size, d_lvr.size)
        if n_obs == 0:
            continue
        sigma = sigma[:n_obs]
        d_fee = d_fee[:n_obs]
        d_lvr = d_lvr[:n_obs]
        bin_idx = np.digitize(sigma, edges[1:-1], right=True)
        for bin_id in range(n_bins):
            mask = bin_idx == bin_id
            if not np.any(mask):
                continue
            fee_values = d_fee[mask]
            lvr_values = d_lvr[mask]
            fee_sum[bin_id] += float(np.sum(fee_values))
            lvr_sum[bin_id] += float(np.sum(lvr_values))
            valid = np.abs(fee_values) > 1e-12
            if not np.any(valid):
                continue
            ratios = lvr_values[valid] / fee_values[valid]
            ratio_sum[bin_id] += float(np.sum(ratios))
            ratio_sumsq[bin_id] += float(np.sum(np.square(ratios)))
            ratio_count[bin_id] += int(ratios.size)

    mean_ratio: List[float] = []
    se_ratio: List[float] = []
    for bin_id in range(n_bins):
        total_fee = float(fee_sum[bin_id])
        total_lvr = float(lvr_sum[bin_id])
        mean_ratio.append(total_lvr / total_fee if abs(total_fee) > 1e-12 else float("nan"))
        count = int(ratio_count[bin_id])
        if count <= 1:
            se_ratio.append(0.0)
            continue
        mean_valid = ratio_sum[bin_id] / count
        variance = (ratio_sumsq[bin_id] - count * mean_valid * mean_valid) / (count - 1)
        se_ratio.append(float(np.sqrt(max(variance, 0.0)) / np.sqrt(count)))

    return _build_ratio_figure(
        cohort=cohort,
        n_bins=n_bins,
        edges=edges,
        mean_ratio=mean_ratio,
        se_ratio=se_ratio,
        bin_labels=bin_labels,
    )
