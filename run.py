"""
Main simulation runner for the ABM model.
X (token0) is like ETH and Y  is like USDC.
"""
# from __future__ import annotations

import argparse
import math
import os
import random
import inspect
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Callable, Set
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_STATIC_WARNING_EMITTED = False
_DEFAULT_GRID_STYLE = dict(showgrid=True, gridcolor="#e1e1e1", gridwidth=1)


def save_plotly_figure(
    fig: go.Figure,
    png_path: Path,
    html_path: Path,
    source: str = "plot",
    *,
    width: int = 1400,
    height: int = 900,
    scale: float = 1.0,
) -> None:
    global PLOTLY_STATIC_WARNING_EMITTED
    """Persist a Plotly figure as both HTML and PNG (if Kaleido is available)."""
    fig.update_xaxes(**_DEFAULT_GRID_STYLE)
    fig.update_yaxes(**_DEFAULT_GRID_STYLE)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=scale)
    except Exception as exc:  # pragma: no cover - depends on kaleido availability
        if not PLOTLY_STATIC_WARNING_EMITTED:
            print(f"[{source}] Warning: could not export Plotly PNGs ({exc})")
            PLOTLY_STATIC_WARNING_EMITTED = True


def plotting_results(
    *,
    results_root: Path,
    pid_str: str,
    fee_mode: str,
    passive_lp_share: float,
    p_jit: float,
    skip_step: int,
    sigma_panel: bool,
    block_time: int,
    steps: np.ndarray,
    P_series: np.ndarray,
    M_series: np.ndarray,
    X_active_end: np.ndarray,
    Y_active_end: np.ndarray,
    band_lo_target: np.ndarray,
    band_hi_target: np.ndarray,
    L_end: np.ndarray,
    L_pre_step: np.ndarray,
    L_pre_trader: np.ndarray,
    L_pre_arb_eff: np.ndarray,
    jiter_wealth_series: np.ndarray,
    jiter_pnl_series: np.ndarray,
    arb_pnl_cum: np.ndarray,
    sr_pnl_cum: np.ndarray,
    noise_pnl_cum: np.ndarray,
    lp_pnl_active_series: np.ndarray,
    lp_pnl_passive_series: np.ndarray,
    lp_unhedged_active_series: np.ndarray,
    lp_unhedged_passive_series: np.ndarray,
    lp_fee_value_total_series: np.ndarray,
    lp_lvr_total_series: np.ndarray,
    dex_notional_y_series: np.ndarray,
    fee_series: List[float],
    fee_sigma_series: np.ndarray,
    fee_basis_ticks_series: np.ndarray,
    fee_signal_series: np.ndarray,
    cex_sigma_series: np.ndarray,
    arb_y_series: List[float],
    sr_y_series: np.ndarray,
    noise_y_series: np.ndarray,
    w_ticks_series: List[int],
    w_unclipped_series: List[float],
    w_noise_series: List[float],
    smart_activity_cum: np.ndarray,
    noise_activity_cum: np.ndarray,
    lp_active_activity_cum: np.ndarray,
    lp_passive_activity_cum: np.ndarray,
    arb_activity_cum: np.ndarray,
    jiter_activity_cum: np.ndarray,
    sr_dex_share_steps: List[int],
    sr_dex_share_series: List[float],
    micro_steps: List[int],
    M_micro: List[float],
    P_micro: List[float],
    micro_valid_steps: List[int],
    micro_valid_prices: List[float],
    mint_steps: List[int],
    mint_sizes: List[float],
    mint_is_passive: List[bool],
    mint_is_jiter: List[bool],
    burn_steps: List[int],
    burn_sizes: List[float],
    burn_is_passive: List[bool],
    burn_is_jiter: List[bool],
    smart_router_enabled: bool,
    noise_trader_enabled: bool,
    lp_active_enabled: bool,
    lp_passive_enabled: bool,
    jiter_enabled: bool,
    max_lag_blocks: int = 15,
    max_lag_micro: int = 15,
) -> None:
    """
    Save Plotly figures summarizing a simulation run (including ACF diagnostics).

    Parameters
    ----------
    results_root
        Root folder where `png/` and `html/` subfolders are created.
    pid_str
        Process id string used in plot filenames.
    fee_mode
        Fee controller mode label used in plot filenames.
    passive_lp_share
        Passive LP share (for plot filenames and titles).
    p_jit
        JIT LP arrival probability (for plot filenames and titles).
    skip_step
        Number of initial blocks to skip in visualizations.
    sigma_panel
        Whether to include a CEX sigma panel (regime/noisy-sine/Heston modes).
    block_time
        Number of micro steps per block (used for micro/block ACF labeling).
    steps, P_series, M_series, X_active_end, Y_active_end, band_lo_target, band_hi_target, L_end, L_pre_step, L_pre_trader, L_pre_arb_eff
        Block-level series (arrays) produced by the simulation.
    jiter_wealth_series, jiter_pnl_series, arb_pnl_cum, sr_pnl_cum, noise_pnl_cum
        Block-level PnL/wealth series (arrays) produced by the simulation.
    lp_pnl_active_series, lp_pnl_passive_series, lp_unhedged_active_series, lp_unhedged_passive_series
        LP PnL series (arrays) produced by the simulation.
    lp_fee_value_total_series, lp_lvr_total_series, dex_notional_y_series
        LP fee value, LP LVR (both cumulative, token1 value), and DEX notional (absolute token1 volume) per block.
    fee_series, fee_sigma_series, fee_basis_ticks_series, fee_signal_series, cex_sigma_series
        Fee controller diagnostic series (arrays/lists) produced by the simulation.
    arb_y_series, sr_y_series, noise_y_series
        Notional series for arbitrageur / smart router / noise trader.
    w_ticks_series, w_unclipped_series, w_noise_series
        LP width rule diagnostics (ticks, unclipped, and noise components).
    smart_activity_cum, noise_activity_cum, lp_active_activity_cum, lp_passive_activity_cum, arb_activity_cum, jiter_activity_cum
        Cumulative signed activity series.
    sr_dex_share_steps, sr_dex_share_series
        Smart-router DEX share over rolling windows.
    micro_steps, M_micro, P_micro, micro_valid_steps, micro_valid_prices
        Micro-time series used to visualize within-block dynamics.
    mint_steps, mint_sizes, mint_is_passive, mint_is_jiter, burn_steps, burn_sizes, burn_is_passive, burn_is_jiter
        LP event series used to build ΔL plots by cohort.
    smart_router_enabled, noise_trader_enabled, lp_active_enabled, lp_passive_enabled, jiter_enabled
        Flags controlling which traces to include in multi-agent panels.
    max_lag_blocks, max_lag_micro
        Maximum lags for the block-level and micro-level log-return ACF plots.

    Returns
    -------
    None

	    Notes
	    -----
	    - "Block" prices refer to the end-of-block DEX price series (`P_series`).
	    - "Micro" prices refer to the within-block, end-of-micro-step series (`P_micro`):
	      one point per micro-step diffusion, with the final micro-step in each block
	      reflecting the cumulative mempool execution effects (and any CEX impact).
	    - ACF is computed on log-returns after applying `skip_step` to blocks and the
	      corresponding `skip_step * block_time` micro-steps.

    Examples
    --------
    >>> import numpy as np
    >>> from pathlib import Path
    >>> plotting_results(
    ...     results_root=Path("abm_results/demo"),
    ...     pid_str="123",
    ...     fee_mode="static",
    ...     passive_lp_share=0.5,
    ...     p_jit=0.0,
    ...     skip_step=0,
    ...     sigma_panel=False,
    ...     block_time=5,
    ...     steps=np.arange(3),
    ...     P_series=np.array([1.0, 1.01, 1.00]),
    ...     M_series=np.array([1.0, 1.02, 0.99]),
    ...     X_active_end=np.zeros(3),
    ...     Y_active_end=np.zeros(3),
    ...     band_lo_target=np.zeros(3),
    ...     band_hi_target=np.zeros(3),
    ...     L_end=np.zeros(3),
    ...     L_pre_step=np.zeros(3),
    ...     L_pre_trader=np.zeros(3),
    ...     L_pre_arb_eff=np.zeros(3),
    ...     jiter_wealth_series=np.zeros(3),
    ...     jiter_pnl_series=np.zeros(3),
    ...     arb_pnl_cum=np.zeros(3),
    ...     sr_pnl_cum=np.zeros(3),
    ...     noise_pnl_cum=np.zeros(3),
    ...     lp_pnl_active_series=np.zeros(3),
    ...     lp_pnl_passive_series=np.zeros(3),
    ...     lp_unhedged_active_series=np.zeros(3),
    ...     lp_unhedged_passive_series=np.zeros(3),
    ...     lp_fee_value_total_series=np.zeros(3),
    ...     lp_lvr_total_series=np.zeros(3),
    ...     dex_notional_y_series=np.zeros(3),
    ...     fee_series=[0.0005, 0.0005, 0.0005],
    ...     fee_sigma_series=np.zeros(3),
    ...     fee_basis_ticks_series=np.zeros(3),
    ...     fee_signal_series=np.zeros(3),
    ...     cex_sigma_series=np.zeros(3),
    ...     arb_y_series=[0.0, 0.0, 0.0],
    ...     sr_y_series=np.zeros(3),
    ...     noise_y_series=np.zeros(3),
    ...     w_ticks_series=[],
    ...     w_unclipped_series=[],
    ...     w_noise_series=[],
    ...     smart_activity_cum=np.zeros(3),
    ...     noise_activity_cum=np.zeros(3),
    ...     lp_active_activity_cum=np.zeros(3),
    ...     lp_passive_activity_cum=np.zeros(3),
    ...     arb_activity_cum=np.zeros(3),
    ...     jiter_activity_cum=np.zeros(3),
    ...     sr_dex_share_steps=[],
    ...     sr_dex_share_series=[],
    ...     micro_steps=[],
    ...     M_micro=[],
    ...     P_micro=[],
    ...     micro_valid_steps=[],
    ...     micro_valid_prices=[],
    ...     mint_steps=[],
    ...     mint_sizes=[],
    ...     mint_is_passive=[],
    ...     mint_is_jiter=[],
    ...     burn_steps=[],
    ...     burn_sizes=[],
    ...     burn_is_passive=[],
    ...     burn_is_jiter=[],
    ...     smart_router_enabled=False,
    ...     noise_trader_enabled=False,
    ...     lp_active_enabled=False,
    ...     lp_passive_enabled=False,
    ...     jiter_enabled=False,
    ... )
    """
    # --- Visualization skip window ---
    s0 = max(0, int(skip_step))
    steps_v = steps[s0:]
    P_series_v = P_series[s0:]
    M_series_v = M_series[s0:]
    X_active_end_v = X_active_end[s0:]
    Y_active_end_v = Y_active_end[s0:]
    band_lo_target_v = band_lo_target[s0:]
    band_hi_target_v = band_hi_target[s0:]
    L_end_v = L_end[s0:]
    L_pre_step_v = L_pre_step[s0:]
    L_pre_trader_v = L_pre_trader[s0:]
    L_pre_arb_eff_v = L_pre_arb_eff[s0:]
    jiter_wealth_series_v = jiter_wealth_series[s0:]
    jiter_pnl_series_v = jiter_pnl_series[s0:]
    arb_pnl_cum_v = arb_pnl_cum[s0:]
    sr_pnl_cum_v = sr_pnl_cum[s0:]
    noise_pnl_cum_v = noise_pnl_cum[s0:]
    lp_pnl_active_series_v = lp_pnl_active_series[s0:]
    lp_pnl_passive_series_v = lp_pnl_passive_series[s0:]
    lp_unhedged_active_series_v = lp_unhedged_active_series[s0:]
    lp_unhedged_passive_series_v = lp_unhedged_passive_series[s0:]
    lp_fee_value_total_series_v = lp_fee_value_total_series[s0:]
    lp_lvr_total_series_v = lp_lvr_total_series[s0:]
    dex_notional_y_series_v = dex_notional_y_series[s0:]
    fee_series_v = fee_series[s0:]
    fee_sigma_series_v = fee_sigma_series[s0:]
    fee_basis_ticks_series_v = fee_basis_ticks_series[s0:]
    fee_signal_series_v = fee_signal_series[s0:]
    cex_sigma_series_v = cex_sigma_series[s0:]
    arb_y_v = np.array(arb_y_series)[s0:]
    sr_y_v = sr_y_series[s0:]
    noise_y_v = noise_y_series[s0:]
    w_ticks_series_v = np.array(w_ticks_series)[s0:] if w_ticks_series else np.array([])
    w_unclipped_series_v = np.array(w_unclipped_series)[s0:] if w_unclipped_series else np.array([])
    w_noise_series_v = np.array(w_noise_series)[s0:] if w_noise_series else np.array([])
    smart_activity_cum_v = smart_activity_cum[s0:]
    noise_activity_cum_v = noise_activity_cum[s0:]
    lp_active_activity_cum_v = lp_active_activity_cum[s0:]
    lp_passive_activity_cum_v = lp_passive_activity_cum[s0:]
    arb_activity_cum_v = arb_activity_cum[s0:]
    jiter_activity_cum_v = jiter_activity_cum[s0:]

    # ΔL per step (split by LP type)
    mint_step_sum_passive = np.zeros_like(P_series)
    mint_step_sum_active = np.zeros_like(P_series)
    n_steps = len(P_series)
    for s, L, is_passive, is_jiter in zip(mint_steps, mint_sizes, mint_is_passive, mint_is_jiter):
        if 0 <= s < n_steps:
            if is_jiter:
                continue
            target = mint_step_sum_passive if is_passive else mint_step_sum_active
            target[s] += L
    burn_step_sum_passive = np.zeros_like(P_series)
    burn_step_sum_active = np.zeros_like(P_series)
    for s, L, is_passive, is_jiter in zip(burn_steps, burn_sizes, burn_is_passive, burn_is_jiter):
        if 0 <= s < n_steps:
            if is_jiter:
                continue
            target = burn_step_sum_passive if is_passive else burn_step_sum_active
            target[s] += L

    mint_step_sum_passive_v = mint_step_sum_passive[s0:]
    mint_step_sum_active_v = mint_step_sum_active[s0:]
    burn_step_sum_passive_v = burn_step_sum_passive[s0:]
    burn_step_sum_active_v = burn_step_sum_active[s0:]

    _out_dir = results_root
    _png_dir = _out_dir / "png"
    _html_dir = _out_dir / "html"
    _png_dir.mkdir(parents=True, exist_ok=True)
    _html_dir.mkdir(parents=True, exist_ok=True)
    # Include key scenario parameters in filenames for clarity.
    _prefix = f"{pid_str}_{fee_mode}_LPpassiveshare{passive_lp_share}_pjit{p_jit}"

    total_steps = max(1, len(steps) - s0)

    def _save_plotly(name: str, fig: go.Figure, *, width: int = 1400, height: int = 900) -> None:
        suffix = f"{name}_steps{total_steps}"
        save_plotly_figure(
            fig,
            _png_dir / f"{_prefix}_{suffix}.png",
            _html_dir / f"{_prefix}_{suffix}.html",
            "simulate",
            width=width,
            height=height,
        )

    steps_list = steps_v.tolist()

    def _finite(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        return arr[np.isfinite(arr)]

    def _finite_nonzero(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr) & (arr != 0)
        return arr[mask]

    def _log_returns(prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        mask = np.isfinite(prices) & (prices > 0)
        prices = prices[mask]
        if prices.size < 2:
            return np.array([], dtype=float)
        return np.diff(np.log(prices))

    def _acf(returns: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if returns.size < 2 or max_lag <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        max_lag_eff = min(int(max_lag), int(returns.size - 1))
        lags = np.arange(1, max_lag_eff + 1, dtype=int)
        acf_vals = np.empty_like(lags, dtype=float)
        for i, lag in enumerate(lags):
            a = returns[:-lag]
            b = returns[lag:]
            if a.size < 2:
                acf_vals[i] = np.nan
                continue
            if np.std(a) < 1e-18 or np.std(b) < 1e-18:
                acf_vals[i] = np.nan
                continue
            acf_vals[i] = float(np.corrcoef(a, b)[0, 1])
        return lags, acf_vals

    # ----- 1) Price panel -----
    cex_returns_v = _log_returns(M_series_v)
    dex_returns_v = _log_returns(P_series_v)
    fig1 = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=("", "CEX log-returns", "DEX log-returns", "DEX log-returns"),
    )
    fig1.add_trace(
        go.Scatter(
            x=steps_list,
            y=band_lo_target_v,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=steps_list,
            y=band_hi_target_v,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(180,180,180,0.35)",
            line=dict(width=0),
            name="No-arb fee band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=steps_list,
            y=P_series_v,
            mode="lines",
            name="DEX price Pₜ",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(
            x=steps_list,
            y=M_series_v,
            mode="lines",
            name="CEX price mₜ",
            line=dict(width=1.6, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Histogram(
            x=_finite(cex_returns_v),
            nbinsx=60,
            marker_color="#1f77b4",
            opacity=0.85,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig1.add_trace(
        go.Histogram(
            x=_finite(dex_returns_v),
            nbinsx=60,
            marker_color="#ff7f0e",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig1.update_layout(
        template="plotly_white",
        title="CEX vs DEX Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig1.update_xaxes(title_text="Block", row=1, col=1)
    fig1.update_yaxes(title_text="Price", row=1, col=1)
    fig1.update_xaxes(title_text="Log-return", row=1, col=2)
    fig1.update_xaxes(title_text="Log-return", row=2, col=2)
    fig1.update_yaxes(title_text="Count", type="log", row=1, col=2)
    fig1.update_yaxes(title_text="Count", type="log", row=2, col=2)
    _save_plotly("1_price", fig1)

    # ----- 1b) Micro-time price panel -----
    if len(M_micro) == len(P_micro) == len(micro_steps) and len(micro_steps) > 0:
        micro_steps_arr = np.asarray(micro_steps, dtype=int)
        M_micro_arr = np.asarray(M_micro, dtype=float)
        P_micro_arr = np.asarray(P_micro, dtype=float)

        # Align micro-time visualization with the burn-in skip used for block plots.
        # We re-index micro steps to start at 0 for readability.
        micro_start_step = max(0, int(s0)) * max(1, int(block_time))
        start_idx = 0
        if micro_start_step > 0:
            start_idx = int(np.searchsorted(micro_steps_arr, micro_start_step, side="left"))
        if start_idx >= micro_steps_arr.size:
            start_idx = 0

        micro_base = int(micro_steps_arr[start_idx]) if micro_steps_arr.size > 0 else 0
        micro_steps_plot = micro_steps_arr[start_idx:] - micro_base
        M_micro_plot = M_micro_arr[start_idx:]
        P_micro_plot = P_micro_arr[start_idx:]
        cex_returns_micro = _log_returns(M_micro_plot)
        dex_returns_micro = _log_returns(P_micro_plot)

        fig1b = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"rowspan": 2}, {}], [None, {}]],
            column_widths=[0.72, 0.28],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            subplot_titles=("", "CEX log-returns", "DEX log-returns", "DEX log-returns"),
        )
        fig1b.add_trace(
            go.Scatter(
                x=micro_steps_plot,
                y=P_micro_plot,
                mode="lines",
                name="DEX price (micro)",
                line=dict(width=1.2),
            ),
            row=1,
            col=1,
        )
        fig1b.add_trace(
            go.Scatter(
                x=micro_steps_plot,
                y=M_micro_plot,
                mode="lines",
                name="CEX price (micro)",
                line=dict(width=1.0, dash="dash"),
            ),
            row=1,
            col=1,
        )
        if micro_valid_steps:
            micro_valid_steps_arr = np.asarray(micro_valid_steps, dtype=int)
            micro_valid_prices_arr = np.asarray(micro_valid_prices, dtype=float)
            mask_valid = micro_valid_steps_arr >= micro_base
            micro_valid_steps_plot = micro_valid_steps_arr[mask_valid] - micro_base
            micro_valid_prices_plot = micro_valid_prices_arr[mask_valid]
            fig1b.add_trace(
                go.Scatter(
                    x=micro_valid_steps_plot,
                    y=micro_valid_prices_plot,
                    mode="markers",
                    name="Validated DEX price",
                    marker=dict(color="#d62728", size=6),
                ),
                row=1,
                col=1,
            )
        fig1b.add_trace(
            go.Histogram(
                x=_finite(cex_returns_micro),
                nbinsx=60,
                marker_color="#1f77b4",
                opacity=0.85,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig1b.add_trace(
            go.Histogram(
                x=_finite(dex_returns_micro),
                nbinsx=60,
                marker_color="#ff7f0e",
                opacity=0.85,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig1b.update_layout(
            template="plotly_white",
            title="Micro-time CEX vs DEX (within blocks)",
        )
        fig1b.update_xaxes(title_text="Micro steps", row=1, col=1)
        fig1b.update_yaxes(title_text="Price", row=1, col=1)
        fig1b.update_xaxes(title_text="Log-return", row=1, col=2)
        fig1b.update_xaxes(title_text="Log-return", row=2, col=2)
        fig1b.update_yaxes(title_text="Count", type="log", row=1, col=2)
        fig1b.update_yaxes(title_text="Count", type="log", row=2, col=2)
        _save_plotly("1b_price_micro", fig1b)

    # ----- 1c) DEX log-return ACF (blocks vs micro) -----
    lags_b, acf_b = _acf(dex_returns_v, max_lag_blocks)
    lags_m = np.array([], dtype=int)
    acf_m = np.array([], dtype=float)
    if len(M_micro) == len(P_micro) == len(micro_steps) and len(micro_steps) > 0:
        micro_steps_arr = np.asarray(micro_steps, dtype=int)
        P_micro_arr = np.asarray(P_micro, dtype=float)
        micro_start_step = max(0, int(s0)) * max(1, int(block_time))
        start_idx = 0
        if micro_start_step > 0:
            start_idx = int(np.searchsorted(micro_steps_arr, micro_start_step, side="left"))
        if 0 < start_idx < P_micro_arr.size:
            P_micro_for_acf = P_micro_arr[start_idx:]
        else:
            P_micro_for_acf = P_micro_arr
        dex_returns_micro_acf = _log_returns(P_micro_for_acf)
        lags_m, acf_m = _acf(dex_returns_micro_acf, max_lag_micro)

    fig1c = make_subplots(
        rows=1,
        cols=1,
        vertical_spacing=0.12,
    )
    
    fig1c.add_trace(go.Bar(x=lags_b, y=acf_b, name="ACF (blocks)"), row=1, col=1)
    # fig1c.add_trace(go.Bar(x=lags_m, y=acf_m, name="ACF (micro)"), row=2, col=1)
    fig1c.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dash"), row=1, col=1)
    # fig1c.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)
    fig1c.update_layout(
        template="plotly_white",
        title=f"DEX Log-Return Autocorrelation",
        showlegend=False,
    )
    fig1c.update_xaxes(title_text="Lag", row=1, col=1)
    # fig1c.update_xaxes(title_text="Lag (micro steps)", row=2, col=1)
    fig1c.update_yaxes(title_text="Autocorrelation", row=1, col=1)
    # fig1c.update_yaxes(title_text="Autocorrelation", row=2, col=1)
    _save_plotly("1c_dex_return_acf", fig1c)

    # ----- 2) Notionals -----
    fig2 = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        subplot_titles=(
            "Notional over time",
            "Smart router",
            "Noise trader",
            "Arbitrageur",
        ),
    )
    fig2.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"), row=1, col=1)
    fig2.add_trace(
        go.Scatter(x=steps_list, y=sr_y_v, mode="lines", name="Smart router", line=dict(width=2, color="#1f77b4")),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=steps_list,
            y=noise_y_v,
            mode="lines",
            name="Noise trader",
            line=dict(dash="dash", color="#ff7f0e"),
        ),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=steps_list,
            y=arb_y_v,
            mode="lines",
            name="Arbitrageur",
            line=dict(dash="dot", color="#2ca02c"),
        ),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Histogram(
            x=_finite_nonzero(sr_y_v),
            nbinsx=60,
            marker_color="#1f77b4",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig2.add_trace(
        go.Histogram(
            x=_finite_nonzero(noise_y_v),
            nbinsx=60,
            marker_color="#ff7f0e",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig2.add_trace(
        go.Histogram(
            x=_finite_nonzero(arb_y_v),
            nbinsx=60,
            marker_color="#2ca02c",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig2.update_layout(
        template="plotly_white",
        title="Trader Notionals",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig2.update_xaxes(title_text="Block", row=1, col=1)
    fig2.update_yaxes(title_text="Notional", row=1, col=1)
    fig2.update_xaxes(title_text="Notional", row=2, col=1)
    fig2.update_xaxes(title_text="Notional", row=2, col=2)
    fig2.update_xaxes(title_text="Notional", row=2, col=3)
    fig2.update_yaxes(title_text="Count", type="log", row=2, col=1)
    fig2.update_yaxes(title_text="Count", type="log", row=2, col=2)
    fig2.update_yaxes(title_text="Count", type="log", row=2, col=3)
    _save_plotly("2_notional", fig2)

    # ----- 2a) Smart-router DEX share (per n blocks) -----
    sr_share_steps_arr = np.asarray(sr_dex_share_steps, dtype=int)
    sr_share_vals_arr = np.asarray(sr_dex_share_series, dtype=float)
    sr_share_mask = sr_share_steps_arr >= s0
    fig2a = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.08,
        subplot_titles=("", ""),
    )
    sr_share_steps_plot = sr_share_steps_arr[sr_share_mask].tolist()
    sr_share_vals_plot = sr_share_vals_arr[sr_share_mask].tolist()
    fig2a.add_trace(
        go.Scatter(
            x=sr_share_steps_plot,
            y=sr_share_vals_plot,
            mode="lines+markers",
            name="DEX share",
        ),
        row=1,
        col=1,
    )
    fig2a.add_trace(
        go.Histogram(
            x=_finite(np.asarray(sr_share_vals_plot, dtype=float)),
            nbinsx=50,
            marker_color="#1f77b4",
            opacity=0.85,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Mark mean DEX share in the histogram for quick reference.
    sr_mean_val = np.mean(_finite(np.asarray(sr_share_vals_plot, dtype=float))) if len(sr_share_vals_plot) > 0 else None
    if sr_mean_val is not None and np.isfinite(sr_mean_val):
        fig2a.add_shape(
            type="line",
            x0=float(sr_mean_val),
            x1=float(sr_mean_val),
            y0=0,
            y1=1,
            xref="x2",
            yref="y2 domain",
            line=dict(color="firebrick", width=2, dash="dash"),
        )
    fig2a.update_layout(
        template="plotly_white",
        title=f"Smart-router DEX Share (DEX/(CEX+DEX))",
    )
    fig2a.update_xaxes(title_text="Block", row=1, col=1)
    fig2a.update_yaxes(title_text="DEX share", range=[0.0, 1.0], row=1, col=1)
    fig2a.update_xaxes(title_text="DEX share", row=1, col=2)
    fig2a.update_yaxes(title_text="Count", type="log", row=1, col=2)
    _save_plotly("2a_smart_router_dex_share", fig2a)

    # ----- 2b) Agent activity (cumulative +/-1) -----
    fig2b = go.Figure()
    fig2b.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
    fig2b.add_trace(
        go.Scatter(
            x=steps_list,
            y=smart_activity_cum_v,
            mode="lines",
            name="Smart router activity",
        )
    )
    fig2b.add_trace(
        go.Scatter(
            x=steps_list,
            y=noise_activity_cum_v,
            mode="lines",
            name="Noise trader activity",
        )
    )
    fig2b.add_trace(
        go.Scatter(
            x=steps_list,
            y=lp_active_activity_cum_v,
            mode="lines",
            name="Active LPs activity",
        )
    )
    fig2b.add_trace(
        go.Scatter(
            x=steps_list,
            y=lp_passive_activity_cum_v,
            mode="lines",
            name="Passive LPs activity",
        )
    )
    fig2b.add_trace(
        go.Scatter(
            x=steps_list,
            y=arb_activity_cum_v,
            mode="lines",
            name="Arbitrageur activity",
        )
    )
    fig2b.add_trace(
        go.Scatter(
            x=steps_list,
            y=jiter_activity_cum_v,
            mode="lines",
            name="Jiter activity",
            line=dict(dash="dash"),
        )
    )
    fig2b.update_layout(
        template="plotly_white",
        title="Agent Activity (cumulative +1 / -1)",
        xaxis_title="Block",
        yaxis_title="Cumulative activity",
    )
    _save_plotly("2b_agent_activity", fig2b)

    # ----- 2c) Agent activity correlation heatmap -----
    activity_labels = ["Smart router", "Noise trader", "Active LPs", "Passive LPs", "Arbitrageur", "Jiter"]
    activity_series = [
        smart_activity_cum_v,
        noise_activity_cum_v,
        lp_active_activity_cum_v,
        lp_passive_activity_cum_v,
        arb_activity_cum_v,
        jiter_activity_cum_v,
    ]
    boot_iters = 100
    alpha = 0.05

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        a_std = a.std()
        b_std = b.std()
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _bootstrap_pvalue(a: np.ndarray, b: np.ndarray, obs_corr: float) -> float:
        n = min(a.size, b.size)
        if n < 2:
            return 1.0
        a_std = a.std()
        b_std = b.std()
        if a_std < 1e-12 or b_std < 1e-12:
            return 1.0
        exceed = 0
        for _ in range(boot_iters):
            idx = np.random.randint(0, n, size=n)
            c_boot = _safe_corr(a[idx], b[idx])
            if abs(c_boot) >= abs(obs_corr):
                exceed += 1
        return exceed / boot_iters

    n_agents = len(activity_series)
    corr_matrix = np.full((n_agents, n_agents), np.nan, dtype=float)
    pval_matrix = np.ones_like(corr_matrix)
    for i, arr_i in enumerate(activity_series):
        for j, arr_j in enumerate(activity_series):
            if j < i:
                # keep lower triangle as NaN (unpopulated)
                continue
            if i == j:
                corr_matrix[i, j] = 1.0 if arr_i.size > 0 else np.nan
                pval_matrix[i, j] = 0.0
            else:
                corr_obs = _safe_corr(arr_i, arr_j)
                corr_matrix[i, j] = corr_obs
                pval_matrix[i, j] = _bootstrap_pvalue(arr_i, arr_j, corr_obs)

    def _cell_text(i: int, j: int) -> str:
        val = corr_matrix[i, j]
        if np.isnan(val):
            return ""
        signif = (i != j) and (pval_matrix[i, j] <= alpha)
        star = "*" if signif else ""
        return f"{val:.3f}{star}"

    fig2c = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=activity_labels,
            y=activity_labels,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=[[_cell_text(i, j) for j in range(len(activity_series))] for i in range(len(activity_series))],
            texttemplate="%{text}",
            textfont=dict(color="black"),
            colorbar=dict(title="Corr"),
        )
    )
    fig2c.update_layout(
        template="plotly_white",
        title="Agent Activity Correlation (cumulative)",
        xaxis_title="Agent",
        yaxis_title="Agent",
    )
    _save_plotly("2c_agent_activity_corr", fig2c)

    # helper for zero-liquidity shading
    def _zero_liquidity_shapes(*, xref: str = "x", yref: str = "paper") -> List[Dict[str, Any]]:
        shapes: List[Dict[str, Any]] = []
        for s_idx, L_val in zip(steps_v, L_end_v):
            if L_val <= 1e-9:
                shapes.append(
                    dict(
                        type="rect",
                        x0=float(s_idx) - 0.5,
                        x1=float(s_idx) + 0.5,
                        y0=0,
                        y1=1,
                        xref=xref,
                        yref=yref,
                        fillcolor="rgba(255,0,0,0.06)",
                        line=dict(width=0),
                    )
                )
        return shapes

    # ----- 3) Liquidity traces -----
    fig3 = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.08,
        subplot_titles=("", ""),
    )
    fig3.add_trace(
        go.Scatter(
            x=steps_list,
            y=L_end_v,
            mode="lines",
            name="Active L (end of step)",
            line=dict(width=1.8),
        ),
        row=1,
        col=1,
    )
    fig3.add_trace(
        go.Histogram(
            x=_finite(L_end_v),
            nbinsx=60,
            marker_color="#1f77b4",
            opacity=0.85,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig3.update_layout(
        template="plotly_white",
        title="Active Liquidity",
        shapes=_zero_liquidity_shapes(xref="x", yref="y domain"),
    )
    fig3.update_xaxes(title_text="Block", row=1, col=1)
    fig3.update_yaxes(title_text="Active L", row=1, col=1)
    fig3.update_xaxes(title_text="Active L", row=1, col=2)
    fig3.update_yaxes(title_text="Count", type="log", row=1, col=2)
    _save_plotly("3_activeL", fig3)

    # ----- 4) L per step (passive vs active) -----
    mint_passive_hist = _finite_nonzero(mint_step_sum_passive_v)
    burn_passive_hist = -_finite_nonzero(burn_step_sum_passive_v)
    mint_active_hist = _finite_nonzero(mint_step_sum_active_v)
    burn_active_hist = -_finite_nonzero(burn_step_sum_active_v)
    fig4 = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=(
            "Passive LPs",
            "Passive LPs",
            "Active LPs",
            "Active LPs",
        ),
    )
    fig4.add_trace(
        go.Bar(x=steps_list, y=mint_step_sum_passive_v, name="Mint / recenter L", marker_color="#6a0dad"),
        row=1,
        col=1,
    )
    fig4.add_trace(
        go.Bar(x=steps_list, y=-burn_step_sum_passive_v, name="Burn L", marker_color="#ff8c00"),
        row=1,
        col=1,
    )
    fig4.add_trace(
        go.Bar(x=steps_list, y=mint_step_sum_active_v, showlegend=False, marker_color="#6a0dad"),
        row=2,
        col=1,
    )
    fig4.add_trace(
        go.Bar(x=steps_list, y=-burn_step_sum_active_v, showlegend=False, marker_color="#ff8c00"),
        row=2,
        col=1,
    )
    fig4.add_trace(
        go.Histogram(
            x=mint_passive_hist,
            nbinsx=60,
            marker_color="#6a0dad",
            opacity=0.75,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig4.add_trace(
        go.Histogram(
            x=burn_passive_hist,
            nbinsx=60,
            marker_color="#ff8c00",
            opacity=0.75,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig4.add_trace(
        go.Histogram(
            x=mint_active_hist,
            nbinsx=60,
            marker_color="#6a0dad",
            opacity=0.75,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig4.add_trace(
        go.Histogram(
            x=burn_active_hist,
            nbinsx=60,
            marker_color="#ff8c00",
            opacity=0.75,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig4.update_layout(
        template="plotly_white",
        title="ΔL per Block",
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig4.update_yaxes(title_text="ΔL per step", row=1, col=1)
    fig4.update_yaxes(title_text="ΔL per step", row=2, col=1)
    fig4.update_xaxes(title_text="Block", row=2, col=1)
    fig4.update_xaxes(title_text="ΔL (mint + / burn -)", row=1, col=2)
    fig4.update_xaxes(title_text="ΔL (mint + / burn -)", row=2, col=2)
    fig4.update_yaxes(title_text="Count", type="log", row=1, col=2)
    fig4.update_yaxes(title_text="Count", type="log", row=2, col=2)
    _save_plotly("4_L_per_step", fig4)

    # ----- 5) Active-band reserves -----
    x_active_value_v = X_active_end_v * P_series_v
    fig5 = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=("", "token0 value (hist)", "", "token1 (hist)"),
    )
    fig5.add_trace(
        go.Scatter(
            x=steps_list,
            y=x_active_value_v,
            mode="lines",
            name="token0 value in active band",
            line=dict(width=1.8),
        ),
        row=1,
        col=1,
    )
    fig5.add_trace(
        go.Scatter(
            x=steps_list,
            y=Y_active_end_v,
            mode="lines",
            name="token1 in active band",
            line=dict(width=1.8),
        ),
        row=1,
        col=1,
    )
    fig5.add_trace(
        go.Histogram(
            x=_finite(x_active_value_v),
            nbinsx=60,
            marker_color="#1f77b4",
            opacity=0.85,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig5.add_trace(
        go.Histogram(
            x=_finite(Y_active_end_v),
            nbinsx=60,
            marker_color="#ff7f0e",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig5.update_layout(
        template="plotly_white",
        title="Active-band Reserves",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        shapes=_zero_liquidity_shapes(xref="x", yref="y domain"),
    )
    fig5.update_xaxes(title_text="Block", row=1, col=1)
    fig5.update_yaxes(title_text="Token1 units", row=1, col=1)
    fig5.update_xaxes(title_text="Token1 units", row=1, col=2)
    fig5.update_xaxes(title_text="Token1 units", row=2, col=2)
    fig5.update_yaxes(title_text="Count", type="log", row=1, col=2)
    fig5.update_yaxes(title_text="Count", type="log", row=2, col=2)
    _save_plotly("5_active_reserves", fig5)

    # ----- 6b) LP mint width signal -----
    if len(w_ticks_series_v) > 0:
        width_baseline_v = w_unclipped_series_v - w_noise_series_v
        fig6b = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.72, 0.28],
            horizontal_spacing=0.08,
            subplot_titles=("", ""),
        )
        fig6b.add_trace(
            go.Scatter(
                x=steps_list,
                y=w_ticks_series_v,
                mode="lines",
                line=dict(width=1.6, dash="dashdot"),
            ),
            row=1,
            col=1,
        )
        fig6b.add_trace(
            go.Histogram(
                x=_finite_nonzero(w_ticks_series_v),
                nbinsx=60,
                marker_color="#1f77b4",
                opacity=0.85,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig6b.update_layout(
            template="plotly_white",
            title="Active LP Mint Width",
        )
        fig6b.update_xaxes(title_text="Block", row=1, col=1)
        fig6b.update_yaxes(title_text="Width (ticks)", row=1, col=1)
        fig6b.update_xaxes(title_text="Width (ticks)", row=1, col=2)
        fig6b.update_yaxes(title_text="Count", type="log", row=1, col=2)
        _save_plotly("6b_mint_width_signal", fig6b)

    # ----- 7) PnL panel -----
    pnl_rows = 2 if sigma_panel else 1
    fig6 = make_subplots(
        rows=pnl_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08 if sigma_panel else 0.1,
        row_heights=[0.7, 0.3] if sigma_panel else None,
        subplot_titles=("Agent PnL", "CEX σ") if sigma_panel else ("Agent PnL",),
    )
    fig6.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"), row=1, col=1)
    if smart_router_enabled:
        fig6.add_trace(
            go.Scatter(x=steps_list, y=sr_pnl_cum_v, mode="lines", name="Smart router PnL"),
            row=1,
            col=1,
        )
    if noise_trader_enabled:
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=noise_pnl_cum_v,
                mode="lines",
                name="Noise trader PnL",
                line=dict(dash="dash"),
            ),
            row=1,
            col=1,
        )
    fig6.add_trace(
        go.Scatter(x=steps_list, y=arb_pnl_cum_v, mode="lines", name="Arbitrageur PnL"),
        row=1,
        col=1,
    )
    if lp_active_enabled:
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=lp_pnl_active_series_v,
                mode="lines",
                name="Active LP hedged",
                line=dict(dash="dash", color="#9467bd"),
            ),
            row=1,
            col=1,
        )
    if jiter_enabled:
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=jiter_pnl_series_v,
                mode="lines",
                name="Jiter hedged net",
                line=dict(width=2, color="#d62728"),
            ),
            row=1,
            col=1,
        )
    if lp_passive_enabled:
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=lp_pnl_passive_series_v,
                mode="lines",
                name="Passive LP hedged",
                line=dict(dash="dot", color="#8c564b"),
            ),
            row=1,
            col=1,
        )
    if sigma_panel and len(cex_sigma_series_v) > 0:
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=cex_sigma_series_v,
                mode="lines",
                name="CEX σ",
                line=dict(width=1.8, color="#2ca02c"),
                line_shape="hv",
            ),
            row=2,
            col=1,
        )
    fig6.update_yaxes(title_text="Token1 value", row=1, col=1)
    if sigma_panel:
        fig6.update_yaxes(title_text="CEX σ", row=2, col=1)
        fig6.update_xaxes(title_text="Block", row=2, col=1)
    else:
        fig6.update_xaxes(title_text="Block", row=1, col=1)
    fig6.update_layout(
        template="plotly_white",
        title="Agent PnL " + (" with CEX σ" if sigma_panel else ""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    _save_plotly("6_pnl", fig6)

    # ----- 7b) Per-block agent PnL (time series + distribution) -----
    def _to_block_pnl(series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        if series.size == 0:
            return series
        # Treat series as an end-of-block cumulative / state variable, and compute
        # per-block increments as first differences. Use prepend=0 to retain the
        # first block's realized change relative to the initial baseline.
        return np.diff(series, prepend=0.0)

    agent_block_pnls: List[Tuple[str, np.ndarray, str]] = []
    if smart_router_enabled:
        agent_block_pnls.append(("Smart router", _to_block_pnl(sr_pnl_cum)[s0:], "#1f77b4"))
    if noise_trader_enabled:
        agent_block_pnls.append(("Noise trader", _to_block_pnl(noise_pnl_cum)[s0:], "#ff7f0e"))
    agent_block_pnls.append(("Arbitrageur", _to_block_pnl(arb_pnl_cum)[s0:], "#2ca02c"))
    if lp_active_enabled:
        agent_block_pnls.append(("Active LP", _to_block_pnl(lp_pnl_active_series)[s0:], "#9467bd"))
    if jiter_enabled:
        agent_block_pnls.append(("Jiter", _to_block_pnl(jiter_pnl_series)[s0:], "#d62728"))
    if lp_passive_enabled:
        agent_block_pnls.append(("Passive LP", _to_block_pnl(lp_pnl_passive_series)[s0:], "#8c564b"))

    if agent_block_pnls:
        n_rows = len(agent_block_pnls)
        fig_height = max(520, 230 * n_rows)
        fig6c = make_subplots(
            rows=n_rows,
            cols=2,
            column_widths=[0.72, 0.28],
            horizontal_spacing=0.08,
            vertical_spacing=0.08 if n_rows > 1 else 0.12,
            row_titles=[label for label, _, _ in agent_block_pnls],
            # column_titles=("Per-block PnL ", "Distribution (count)"),
        )
        for row_i, (label, pnl_block_v, color) in enumerate(agent_block_pnls, start=1):
            fig6c.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"), row=row_i, col=1)
            fig6c.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=pnl_block_v,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=row_i,
                col=1,
            )
            fig6c.add_trace(
                go.Histogram(
                    x=_finite(pnl_block_v),
                    nbinsx=60,
                    marker_color=color,
                    opacity=0.85,
                    showlegend=False,
                ),
                row=row_i,
                col=2,
            )
        fig6c.update_yaxes(title_text="PnL ", col=1)
        fig6c.update_yaxes(title_text="Count", type="log", col=2)
        fig6c.update_xaxes(title_text="Block", row=n_rows, col=1)
        fig6c.update_xaxes(title_text="PnL ", row=n_rows, col=2)
        fig6c.update_layout(
            template="plotly_white",
            title="Per-block Agent PnL ",
            showlegend=False,
            bargap=0.05,
            height=fig_height,
        )
        _save_plotly(
            "6c_pnl_per_block",
            fig6c,
            height=fig_height,
        )

    # ----- 8) Fee panel + controller signal -----
    fig7 = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    if fee_mode in ("volatility_cex", "volatility_dex"):
        secondary_vals_full = fee_sigma_series_v
        secondary_label = "EWMA(σ^2)"
    elif fee_mode == "toxicity":
        secondary_vals_full = fee_basis_ticks_series_v
        secondary_label = "Basis (ticks)"
    elif fee_mode == "lvr_fee_ewma":
        secondary_vals_full = fee_signal_series_v
        secondary_label = "EWMA(dLVR - dFees) / notional"
    else:
        secondary_vals_full = fee_signal_series_v
        secondary_label = "Controller signal"

    # Signals are computed from END-OF-STEP state at t, and the scheduled update is
    # only committed at the START of step t+1. For visualization (signal→fee), plot
    # fee_{t+1} at timestamp t (i.e., shift the fee series back by 1 step).
    if len(steps_list) > 1:
        steps_fee_plot = steps_list[:-1]
        fee_plot = fee_series_v[1:]
        secondary_vals_plot = secondary_vals_full[:-1]
        fee_label = "Fee (applies next step; aligned to signal)"
    else:
        steps_fee_plot = steps_list
        fee_plot = fee_series_v
        secondary_vals_plot = secondary_vals_full
        fee_label = "Fee"

    fig7.add_trace(
        go.Scatter(
            x=steps_fee_plot,
            y=fee_plot,
            mode="lines",
            name=fee_label,
            line=dict(width=1.8),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig7.add_trace(
        go.Scatter(
            x=steps_fee_plot,
            y=secondary_vals_plot,
            mode="lines",
            name=secondary_label,
            line=dict(width=1.2, dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    # Empirical distribution of fees with mean/median/percentile markers
    fee_series_arr = np.asarray(fee_series_v, dtype=float)
    fee_has_data = fee_series_arr.size > 0
    fee_mean = float(np.mean(fee_series_arr)) if fee_has_data else 0.0
    fee_median = float(np.median(fee_series_arr)) if fee_has_data else 0.0
    percentile_levels = [5, 25, 75, 95]
    fee_percentiles = (
        [(p, float(np.percentile(fee_series_arr, p))) for p in percentile_levels]
        if fee_has_data
        else []
    )
    percentile_styles = [
        dict(color="#6b7280", dash="dot"),
        dict(color="#9ca3af", dash="dash"),
        dict(color="#4b5563", dash="dashdot"),
        dict(color="#374151", dash="longdash"),
    ]
    hist_xref = "x2"
    hist_yref = "y3 domain"
    fig7.add_trace(
        go.Histogram(
            x=fee_series_v,
            name="Fee distribution",
            marker_color="#1f77b4",
            opacity=0.75,
        ),
        row=2,
        col=1,
    )
    if fee_has_data:
        fig7.add_shape(
            type="line",
            x0=fee_mean,
            x1=fee_mean,
            y0=0,
            y1=1,
            xref=hist_xref,
            yref=hist_yref,
            line=dict(color="firebrick", width=2, dash="dash"),
        )
        fig7.add_shape(
            type="line",
            x0=fee_median,
            x1=fee_median,
            y0=0,
            y1=1,
            xref=hist_xref,
            yref=hist_yref,
            line=dict(color="black", width=2, dash="dot"),
        )
        # legend handles for mean/median/percentiles
        fig7.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"Mean = {fee_mean:.5f}",
                line=dict(color="firebrick", width=2, dash="dash"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        fig7.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"Median = {fee_median:.5f}",
                line=dict(color="black", width=2, dash="dot"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        for (p_level, p_value), style in zip(fee_percentiles, percentile_styles):
            fig7.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"P{p_level:02d} = {p_value:.5f}",
                    line=dict(color=style["color"], width=1.6, dash=style["dash"]),
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
    fig7.update_layout(
        template="plotly_white",
        title="Fee & Controller Signal (fee aligned to prior step's signal)",
        xaxis_title="Block",
        xaxis2_title="Fee",
        yaxis2_title="Count",
    )
    fig7.update_yaxes(title_text="Fee", secondary_y=False)
    fig7.update_yaxes(title_text=secondary_label, secondary_y=True)
    _save_plotly("7_fee", fig7)

    # ----- 8) Normalized LVR diagnostics (per-block) -----
    # Rationale: raw ΔLVR (token1/block) mixes "toxicity" with activity effects (volume, block aggregation).
    # Normalizing by DEX notional and by fees yields (i) LVR-per-volume and (ii) fee-coverage metrics that
    # are more comparable across fee modes / block sizes.
    eps = 1e-18
    d_lvr_total_v = np.diff(lp_lvr_total_series, prepend=0.0)[s0:]
    d_fee_value_total_v = np.diff(lp_fee_value_total_series, prepend=0.0)[s0:]

    lvr_per_notional_bps = np.full_like(d_lvr_total_v, np.nan, dtype=float)
    mask_notional = np.isfinite(d_lvr_total_v) & np.isfinite(dex_notional_y_series_v) & (dex_notional_y_series_v > eps)
    lvr_per_notional_bps[mask_notional] = 1e4 * d_lvr_total_v[mask_notional] / dex_notional_y_series_v[mask_notional]

    lvr_over_fees = np.full_like(d_lvr_total_v, np.nan, dtype=float)
    # Use only blocks with positive fee-value increment; fee value is marked-to-market and can drop if m_t falls.
    mask_fees = np.isfinite(d_lvr_total_v) & np.isfinite(d_fee_value_total_v) & (d_fee_value_total_v > eps)
    lvr_over_fees[mask_fees] = d_lvr_total_v[mask_fees] / d_fee_value_total_v[mask_fees]

    fig8 = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
        subplot_titles=(
            "LVR / DEX notional (bps) — time series",
            "LVR / Fees (ΔLVR/ΔFees) — time series",
            "LVR / DEX notional (bps) — distribution",
            "LVR / Fees (ΔLVR/ΔFees) — distribution",
        ),
    )
    fig8.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"), row=1, col=1)
    fig8.add_trace(
        go.Scatter(
            x=steps_list,
            y=lvr_per_notional_bps,
            mode="lines",
            name="LVR/notional (bps)",
            line=dict(width=1.6, color="#111827"),
            showlegend=False,
            hovertemplate="t=%{x}<br>bps=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig8.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"), row=1, col=2)
    fig8.add_hline(y=1.0, line=dict(color="#6b7280", width=1, dash="dash"), row=1, col=2)
    fig8.add_trace(
        go.Scatter(
            x=steps_list,
            y=lvr_over_fees,
            mode="lines",
            name="LVR/fees",
            line=dict(width=1.6, color="#111827"),
            showlegend=False,
            hovertemplate="t=%{x}<br>ratio=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig8.add_trace(
        go.Histogram(
            x=_finite(lvr_per_notional_bps),
            nbinsx=80,
            marker_color="#111827",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig8.add_trace(
        go.Histogram(
            x=_finite(lvr_over_fees),
            nbinsx=80,
            marker_color="#111827",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig8.update_xaxes(title_text="Block", row=1, col=1)
    fig8.update_xaxes(title_text="Block", row=1, col=2)
    fig8.update_xaxes(title_text="bps", row=2, col=1)
    fig8.update_xaxes(title_text="ratio", row=2, col=2)
    fig8.update_yaxes(title_text="bps", row=1, col=1)
    fig8.update_yaxes(title_text="ratio", row=1, col=2)
    fig8.update_yaxes(title_text="Count", type="log", row=2, col=1)
    fig8.update_yaxes(title_text="Count", type="log", row=2, col=2)
    fig8.update_layout(
        template="plotly_white",
        title="Normalized LVR diagnostics (per-block)",
        bargap=0.05,
    )
    _save_plotly("8_normalized_lvr", fig8)

# Import from new module structure
from utils import (
    build_empty_pool,
    bootstrap_initial_binomial_hill_sharded,
    minted_amounts_at_S,
    ReferenceMarket,
    EWMA,
    next_numbered_path,
    EPS_LIQ,
    EPS_LIQ2,
    EPS_BOUNDARY,
    EPS_PRICE_CHANGE,
    TICK_LN,
    clamp,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    make_liquidity_gif,
    load_simulation_parameters,
    scenario_output_root,
)
from agents import (
    LPAgent,
    Position,
    lp_token0_exposure,
    lp_wealth_y,
    lp_total_fee_earned_value_y,
    lp_total_position_value_y,
)
from collections import defaultdict

from uniswapv3_pool import V3Pool


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class TraderStepAccumulator:
    notional_y: float = 0.0
    pnl: float = 0.0
    execs: int = 0
    dx_in: float = 0.0
    dx_out: float = 0.0
    dy_in: float = 0.0
    dy_out: float = 0.0
    # Track realized PnL from CEX trades separately (already settled at execution)
    realized_pnl_cex: float = 0.0

    def record_swap(
        self,
        *,
        dx_in: float = 0.0,
        dx_out: float = 0.0,
        dy_in: float = 0.0,
        dy_out: float = 0.0,
    ) -> None:
        """Track token flows for later PnL settlement (DEX trades only)."""
        self.dx_in += dx_in
        self.dx_out += dx_out
        self.dy_in += dy_in
        self.dy_out += dy_out

    def record_cex_trade_pnl(self, pnl: float) -> None:
        """
        Record realized PnL from a CEX trade that is already settled at execution price.
        CEX trades are fair exchanges at the current CEX price, so PnL is typically 0.
        This bypasses the settle() revaluation.
        """
        self.realized_pnl_cex += pnl

    def settle(self, m_settle: float) -> None:
        """
        Value accumulated DEX flows versus the provided CEX price.
        Positive result means net token1 profit.
        Final PnL includes both DEX settlement and already-realized CEX PnL.
        """
        dex_pnl = (self.dy_out - self.dy_in) + (self.dx_out - self.dx_in) * m_settle
        self.pnl = dex_pnl + self.realized_pnl_cex

class _NullList(list):
    """
    Lightweight sink used in light_mode to avoid collecting data we won't use.
    Behaves like an always-empty list; append/extend are no-ops.
    """

    def append(self, *args, **kwargs):  # type: ignore[override]
        return None

    def extend(self, *args, **kwargs):  # type: ignore[override]
        return None

    def __iadd__(self, other):  # type: ignore[override]
        return self

def simulate(
    # === Core simulation parameters ===
    config_name: str,
    block_time: int,
    T: int,
    seed: int,
    liquidity_for_gif: bool,
    light_mode: bool,
    
    # === Market parameters ===
    cex_mu: float,
    cex_sigma: float,
    
    # === Trade flow parameters ===
    smart_trades_per_block: float,
    noise_trades_per_block: float,
    
    # === LP population parameters ===
    N_LP: int,
    passive_lp_share: float,
    tau: int,
    narrow_mints_per_block: float,
    passive_mints_per_block: float,
    passive_burns_per_block: float,
    
    # === LP width parameters ===
    w_min_ticks: int,
    w_max_ticks: int,
    basis_half_life: int,   # steps
    slope_s: float,        # ticks per (basis-in-ticks)
    binom_n: int,
    binom_p: float,
    
    # === Trader parameters ===
    trader_mean: float,
    trader_sigma: float,
    theta_T: float,
    slippage_tolerance: float,
    passive_width_pct: Optional[float],  # total width percentage (± half around price) for passive LPs
    passive_width_ticks: Optional[int],
    
    # === LP behavior parameters ===
    mint_mu: float,
    mint_sigma: float,
    theta_TP: float,
    theta_SL: float,
    k_out_min: int,
    k_out_max: int,
    
    # === Initial conditions ===
    initial_binom_N: int,
    initial_total_L: float,
    
    # === Fee controller parameters ===
    fee_mode: str,      # "static" | "volatility_cex" | "volatility_dex" | "toxicity" | "lvr_fee_ewma"
    f0: float,             # initial fee (and static fee level), e.g. 30 bps
    f_min: float,         # 5 bps
    f_max: float,           # 200 bps safety cap
    fee_half_life: int,       # EWMA half-life (steps) for signals
    k_sigma: float,         # adds ~k_sigma * EWMA(|logret|) to fee
    k_basis: float,         # fee per tick of dislocation (basis in ticks)
    fee_step_bps_min: float, # do not change fee unless ≥ 0.5 bps move
    fee_step_bps_max: float, # max step per update (bps)
    fee_cooldown: int,         # blocks between fee changes (hysteresis)
    k_lvr: float = 0.0,     # feedback gain for "lvr_fee_ewma" (dimensionless)
    
    # === Cost parameters ===
    flash_loan_fee: float = 0.0,  # percentage cost on arbitrage notional (e.g., 0.0005 = 5 bps)
    
    # === Jiter (JIT LP searcher) parameters ===
    jit_flash_loan_fee: float = 0.0,  # percentage cost on JIT principal value , per mint
    p_jit: float = 0.0,           # Bernoulli arrival probability per block
    N_jit: int = 0,               # target up to top-N swaps by input amount
    liquidity_perc_jit: float = 0.0,  # target share of active liquidity (0-1)
    
    # === CEX volatility modes ===
    cex_sigma_mode: str = "static",    # "static" | "regime" | "noisy_sine" | "heston"
    cex_sigma_low: Optional[float] = None,
    cex_sigma_high: Optional[float] = None,
    cex_sigma_p_LL: float = 1.0,
    cex_sigma_p_HH: float = 1.0,
    cex_sigma_regime_init: str = "L",
    cex_sigma_sine_period: int = 10_000,
    cex_sigma_sine_amp: Optional[float] = None,
    cex_sigma_sine_noise: float = 0.0,
    cex_sigma_floor: float = 0.0,
    cex_sigma_sine_phase: float = 0.0,
    
    # === CEX Heston-like volatility parameters ===
    cex_heston_kappa: Optional[float] = None,
    cex_heston_theta: Optional[float] = None,
    cex_heston_sigma_v: Optional[float] = None,
    cex_heston_rho: Optional[float] = None,
    cex_heston_v0: Optional[float] = None,
    
    # === Output and visualization ===
    visualize: bool = True,
    skip_step: int = 100,
    n_block_SR_ratio: int = 100,
    results_root: Optional[str | Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:

    valid_fee_modes = {"static", "volatility_cex", "volatility_dex", "toxicity", "lvr_fee_ewma"}
    if fee_mode not in valid_fee_modes:
        raise ValueError(f"Invalid fee_mode '{fee_mode}'. Expected one of {sorted(valid_fee_modes)}.")
    if k_out_min <= 0 or k_out_max <= 0:
        raise ValueError("k_out_min and k_out_max must be positive integers.")
    if k_out_min > k_out_max:
        raise ValueError("k_out_min cannot exceed k_out_max.")

    p_jit = clamp(p_jit, 0.0, 1.0)
    N_jit = max(0, int(N_jit))
    liquidity_perc_jit = clamp(liquidity_perc_jit, 0.0, 0.999999)
    slippage_tolerance = clamp(slippage_tolerance, 0.0, 1.0)
    try:
        block_time = int(block_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"block_time must be an integer > 1: got {block_time!r}") from exc
    if block_time <= 1:
        raise ValueError("block_time must be > 1 for mempool execution mode.")

    try:
        n_block_SR_ratio = int(n_block_SR_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n_block_SR_ratio must be a positive integer: got {n_block_SR_ratio!r}") from exc
    if n_block_SR_ratio <= 0:
        raise ValueError(f"n_block_SR_ratio must be a positive integer: got {n_block_SR_ratio}")

    if passive_width_pct is not None:
        try:
            passive_width_pct = float(passive_width_pct)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"passive_width_pct must be a number: got {passive_width_pct!r}") from exc
        if passive_width_pct > 100.0:
            print(f"passive_width_pct was set to {passive_width_pct}. Clamping it to 100%")
            passive_width_pct = 100.0

    try:
        flash_loan_fee = float(flash_loan_fee)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"flash_loan_fee must be a non-negative number: got {flash_loan_fee!r}") from exc
    if flash_loan_fee < 0.0:
        raise ValueError(f"flash_loan_fee must be >= 0.0: got {flash_loan_fee}")
    flash_loan_mult = 1.0 + flash_loan_fee

    try:
        jit_flash_loan_fee = float(jit_flash_loan_fee)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"jit_flash_loan_fee must be a non-negative number: got {jit_flash_loan_fee!r}") from exc
    if jit_flash_loan_fee < 0.0:
        raise ValueError(f"jit_flash_loan_fee must be >= 0.0: got {jit_flash_loan_fee}")

    initial_params = dict(locals())

    def _vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if light_mode:
        visualize = False
        verbose = False

    # Normalize output root for this simulation (logs + plots).
    if results_root is None:
        results_root_path = Path("abm_results")
    else:
        results_root_path = Path(results_root)

    np.random.seed(seed) # for numpy
    random.seed(seed) # for built-in python random functions

    passive_share = max(0.0, min(1.0, passive_lp_share))
    
    # Trade arrivals are modeled as a Poisson process per micro-step.
    # Each block has B micro-steps, so lambda_micro = trades_per_block / B.
    B = block_time
    smart_trades_per_block = max(0.0, float(smart_trades_per_block))
    noise_trades_per_block = max(0.0, float(noise_trades_per_block))
    smart_lambda_micro = smart_trades_per_block / B
    noise_lambda_micro = noise_trades_per_block / B
    # Track which cohorts are active for plotting/metrics.
    lp_active_enabled = passive_share < 1.0
    lp_passive_enabled = passive_share > 0.0
    smart_router_enabled = smart_trades_per_block > 0.0
    noise_trader_enabled = noise_trades_per_block > 0.0

    jiter_enabled = p_jit > 0.0 and N_jit > 0 and liquidity_perc_jit > 0.0
    jiter_agent: Optional[LPAgent] = None

    # narrow_agents = max(1e-12, (1.0 - passive_share) * max(1, N_LP))
    # passive_agents = max(1e-12, passive_share * max(1, N_LP))
    narrow_mints_per_block = max(0.0, float(narrow_mints_per_block))
    passive_mints_per_block = max(0.0, float(passive_mints_per_block))
    passive_burns_per_block = max(0.0, float(passive_burns_per_block))

    # --- Log Annualized Volatility for Clarity ---
    # cex_sigma is per-micro-step (1 second).
    # Annualized Volatility = cex_sigma * sqrt(seconds_in_year)
    seconds_per_year = 365 * 24 * 60 * 60
    sigma_mode_norm = (cex_sigma_mode or "static").lower()
    regime_mode = sigma_mode_norm in {"regime", "regime_switch", "regime_switching"}
    noisy_sine_mode = sigma_mode_norm == "noisy_sine"
    heston_mode = sigma_mode_norm == "heston"
    regime_state = "H" if str(cex_sigma_regime_init).upper().startswith("H") else "L"
    sigma_mode_for_ref = (
        "regime"
        if regime_mode
        else ("noisy_sine" if noisy_sine_mode else ("heston" if heston_mode else "static"))
    )

    if heston_mode:
        # Fail fast on missing Heston parameters.
        missing = []
        if cex_heston_kappa is None:
            missing.append("cex_heston_kappa")
        if cex_heston_theta is None:
            missing.append("cex_heston_theta")
        if cex_heston_sigma_v is None:
            missing.append("cex_heston_sigma_v")
        if cex_heston_rho is None:
            missing.append("cex_heston_rho")
        if missing:
            raise ValueError(
                "cex_sigma_mode='heston' requires parameters: " + ", ".join(missing)
            )
        if not (-1.0 <= float(cex_heston_rho) <= 1.0):
            raise ValueError(
                "cex_heston_rho must be in [-1, 1] when cex_sigma_mode='heston'."
            )
        # Determine initial per-step sigma from explicit v0 or cex_sigma.
        if cex_heston_v0 is not None:
            v0 = float(cex_heston_v0)
            if v0 <= 0.0:
                raise ValueError(
                    "cex_heston_v0 must be positive when cex_sigma_mode='heston'."
                )
            sigma_for_ref = math.sqrt(v0)
        else:
            sigma_for_ref = float(cex_sigma)
            if sigma_for_ref <= 0.0:
                raise ValueError(
                    "cex_sigma must be positive when cex_sigma_mode='heston' and cex_heston_v0 is not provided."
                )
        sigma_annualized = sigma_for_ref * math.sqrt(seconds_per_year)
        _vprint(
            f"[CONFIG] Heston cex_sigma_mode: "
            f"sigma0={sigma_for_ref} ({sigma_annualized:.2%} annualized), "
            f"kappa={cex_heston_kappa}, theta={cex_heston_theta}, "
            f"sigma_v={cex_heston_sigma_v}, rho={cex_heston_rho}, v0={cex_heston_v0}"
        )
    if regime_mode:
        if cex_sigma_low is None or cex_sigma_high is None:
            raise ValueError("cex_sigma_low and cex_sigma_high must be set when cex_sigma_mode='regime'.")
        if cex_sigma_high <= cex_sigma_low:
            raise ValueError(f"Require cex_sigma_high > cex_sigma_low (got {cex_sigma_low} >= {cex_sigma_high}).")
        if not (0.0 <= cex_sigma_p_LL <= 1.0) or not (0.0 <= cex_sigma_p_HH <= 1.0):
            raise ValueError("cex_sigma_p_LL and cex_sigma_p_HH must be probabilities in [0, 1].")
        sigma_annualized_low = cex_sigma_low * math.sqrt(seconds_per_year)
        sigma_annualized_high = cex_sigma_high * math.sqrt(seconds_per_year)
        _vprint(
            f"\n[CONFIG] Regime-switching cex_sigma: "
            f"low={cex_sigma_low} ({sigma_annualized_low:.2%} annualized), "
            f"high={cex_sigma_high} ({sigma_annualized_high:.2%} annualized), "
            f"start={regime_state}, p_LL={cex_sigma_p_LL}, p_HH={cex_sigma_p_HH}"
        )
        sigma_for_ref = cex_sigma_low if regime_state == "L" else cex_sigma_high
    else:
        sigma_annualized = cex_sigma * math.sqrt(seconds_per_year)
        _vprint(f"[CONFIG] cex_sigma={cex_sigma} (per 1s step) => Annualized Volatility: {sigma_annualized:.2%}")
        sigma_for_ref = cex_sigma
    _vprint(f"[CONFIG] Initial fee f0: {initial_params.get('f0', 0.0005)*10000:.1f} bps")
    _vprint(f"[CONFIG] LP passive share: {passive_share:.2%}, N_LP={N_LP}, P JIT={p_jit}\nSmart-Router target: {smart_trades_per_block:.2f} Noise trader target: {noise_trades_per_block:.2f}\n Narrow LP mints/block: {narrow_mints_per_block:.2f} Passive LP mints/block: {passive_mints_per_block:.2f} Passive LP burns/block: {passive_burns_per_block:.2f}\n")

    # --- Build pool + reference market + LP agents ----------------------------
    pool, m0 = build_empty_pool()

    def _snap_up_tick(tick: int) -> int:
        '''Snap a tick to the nearest multiple of the tick spacing.'''
        spacing = int(pool.tick_spacing)
        tick_int = int(tick)
        return -((-tick_int) // spacing) * spacing

    def _passive_range_ticks_from_pct(S_now: float) -> Tuple[int, int]:
        '''Compute the mint range from the current price and the passive width percentage.'''
        if passive_width_pct is None:
            raise RuntimeError("Internal error: passive_width_pct is None.")
        half = float(passive_width_pct) / 200.0
        if half <= 0.0 or half >= 1.0:
            raise ValueError(f"passive_width_pct must be in (0, 200): got {passive_width_pct}")
        S_now_f = float(S_now)
        S_low = S_now_f * math.sqrt(1.0 - half)
        S_high = S_now_f * math.sqrt(1.0 + half)
        tick_low_real = math.log(max(S_low, 1e-18) / pool.base_s, pool.g)
        tick_high_real = math.log(max(S_high, 1e-18) / pool.base_s, pool.g)
        lower = pool._snap(int(math.floor(tick_low_real)))
        upper = _snap_up_tick(int(math.ceil(tick_high_real)))
        if upper <= lower:
            upper = lower + pool.tick_spacing
        return lower, upper

    ref = ReferenceMarket(
        m=m0,
        mu=cex_mu,
        sigma=sigma_for_ref,
        kappa=1e-3,
        sigma_mode=sigma_mode_for_ref,
        sigma_low=cex_sigma_low,
        sigma_high=cex_sigma_high,
        p_LL=cex_sigma_p_LL,
        p_HH=cex_sigma_p_HH,
        regime_state=regime_state,
        sigma_sine_amp=cex_sigma_sine_amp,
        sigma_sine_period=cex_sigma_sine_period,
        sigma_sine_noise=cex_sigma_sine_noise,
        sigma_floor=cex_sigma_floor,
        sigma_sine_phase=cex_sigma_sine_phase,
        heston_kappa=cex_heston_kappa,
        heston_theta=cex_heston_theta,
        heston_sigma_v=cex_heston_sigma_v,
        heston_rho=cex_heston_rho,
        heston_v0=cex_heston_v0,
    )

    LPs: List[LPAgent] = []
    # Deterministically assign exactly round((1 - passive_share) * N_LP) active narrow LPs.
    total_lps = max(0, int(N_LP))
    target_active = int(round((1.0 - passive_share) * total_lps))
    target_active = max(0, min(total_lps, target_active))
    indices = list(range(total_lps))
    random.shuffle(indices)
    active_indices = set(indices[:target_active])

    for i in range(N_LP):
        is_narrow = i in active_indices
        is_passive = not is_narrow
        # LP mint/burn arrival is scheduled explicitly in the mempool path
        # using Poisson counts, so per-LP mintProb is not used for targeting.
        mintProb = 0.0
        LPs.append(
            LPAgent(
                id=i,
                mintProb=mintProb,
                is_active_narrow=is_narrow,
                is_passive=is_passive,
            )
        )
        lp = LPs[-1]
        lp.review_rate = 1.0 / max(1, tau)
        lp.next_review = int(np.random.geometric(lp.review_rate))
        lp.cooldown = 0
        lp.can_act = False
        lp.k_out_threshold = random.randint(k_out_min, k_out_max)

    bootstrap_initial_binomial_hill_sharded(
        pool, ref, LPs,
        N=initial_binom_N,
        L_total=initial_total_L,
        num_seed_lps=20,
        seed_lp_id_base=10_000,
        seed_mint_prob=0.0,
        tau=tau,
        plot=False,
        seed_is_passive=True,
    )

    # -------------------------------------------------------------------------
    # LP initial cash inventories (LPs budget in token1 only)
    # -------------------------------------------------------------------------
    # Initialize each strategic (non-seed, non-JIT) LP wallet as an equal share of
    # the *value* of the initial binomial-hill liquidity at the initial price.
    #
    # Strategic LPs hold cash in token1 only. When minting, they convert an
    # appropriate amount into token0 on the CEX (impacting the CEX price) and then
    # deposit the resulting (token0, token1) amounts into the AMM.
    #
    # Note: seed LPs start with their liquidity already deployed as positions, so
    # they begin with wallet_y=0 and can only mint again after earning cash back
    # via burns.
    seed_total_x = 0.0
    seed_total_y = 0.0
    for lp in LPs:
        if not bool(getattr(lp, "is_seed", False)):
            continue
        for pos in getattr(lp, "positions", []):
            seed_total_x += float(pos.amt0_init)
            seed_total_y += float(pos.amt1_init)

    denom_strategic = max(1, int(N_LP))
    initial_seed_value_y = seed_total_x * float(m0) + seed_total_y
    per_lp_wallet_y = initial_seed_value_y / denom_strategic
    for lp in LPs:
        if bool(getattr(lp, "is_seed", False)) or bool(getattr(lp, "is_jiter", False)):
            continue
        lp.wallet_x = 0.0
        lp.wallet_y = float(per_lp_wallet_y)

    # ensure k_out_threshold exists for every LP (including seeds)
    for lp in LPs:
        if not hasattr(lp, "k_out_threshold"):
            lp.k_out_threshold = random.randint(k_out_min, k_out_max)

    if jiter_enabled:
        jiter_agent = LPAgent(
            id=1_000_000,
            mintProb=0.0,
            is_active_narrow=False,
            is_passive=False,
            is_seed=True,  # exclude from cohort aggregates; tracked separately
        )
        jiter_agent.is_jiter = True  # type: ignore[attr-defined]
        jiter_agent.review_rate = 0.0
        jiter_agent.next_review = 0
        jiter_agent.cooldown = 0
        jiter_agent.can_act = False
        jiter_agent.L_budget = float("inf")
        jiter_agent.L_live = 0.0
        jiter_agent.wallet_x = 0.0
        jiter_agent.wallet_y = 0.0
        jiter_agent.flash_fees_paid_y = 0.0  # type: ignore[attr-defined]
        LPs.append(jiter_agent)

    lp_lookup: Dict[int, LPAgent] = {lp.id: lp for lp in LPs}
    # NOTE on determinism:
    # We keep per-tick position buckets as *lists* (in insertion order) rather than sets.
    # Python randomizes hash seeds per process by default, so iterating a set can yield a
    # different order across runs even when RNG seeds are fixed; that can change the
    # floating-point accumulation order in fee allocation and cascade into divergent
    # agent behavior and price paths.
    positions_by_tick: Dict[int, List[Position]] = defaultdict(list)

    def _register_position(pos: Position) -> None:
        slots = tuple(range(pos.lower, pos.upper, pool.tick_spacing))
        pos.tick_slots = slots
        for tick_val in slots:
            positions_by_tick[tick_val].append(pos)

    def _unregister_position(pos: Position) -> None:
        for tick_val in getattr(pos, "tick_slots", ()):
            bucket = positions_by_tick.get(tick_val)
            if not bucket:
                continue
            try:
                bucket.remove(pos)
            except ValueError:
                continue
            if not bucket:
                positions_by_tick.pop(tick_val, None)
        pos.tick_slots = tuple()

    def _assert_active_liquidity_state_fast(label: str) -> None:
        """Lightweight guard: non-negative L_active and price within the active band."""
        if abs(pool.L_active) <= EPS_LIQ2:
            pool.L_active = 0.0
            return

        underflow_tol = 100.0 * EPS_LIQ2
        if pool.L_active < -underflow_tol:
            raise AssertionError(f"L_active underflow ({label}): {pool.L_active}")

        if pool.L_active > EPS_LIQ:
            sa = pool.s_lower()
            sb = pool.s_upper()
            band_scale = max(1.0, abs(sa), abs(sb), abs(pool.S))
            boundary_tol = EPS_BOUNDARY * band_scale
            if pool.S < sa - boundary_tol or pool.S > sb + boundary_tol:
                raise AssertionError(
                    f"Price S={pool.S} outside active band during {label}: tick={pool.tick} band=[{sa},{sb}]"
                )
            if pool.S < sa:
                pool.S = sa
            elif pool.S > sb:
                pool.S = sb

    def _assert_active_liquidity_state_full(label: str) -> None:
        """Full guard: ensure pool.L_active agrees with liquidity_net prefix sums."""
        prefix_L = pool.bidx.active_liquidity_at_tick(pool.tick)

        # If both representations are numerically tiny, snap to zero and skip.
        if abs(pool.L_active) <= EPS_LIQ2 and abs(prefix_L) <= EPS_LIQ2:
            pool.L_active = 0.0
            return

        # Negative active liquidity is a hard invariant violation.
        underflow_tol = 100.0 * EPS_LIQ2
        if pool.L_active < -underflow_tol:
            raise AssertionError(f"L_active underflow ({label}): {pool.L_active}")

        def _mismatch_tolerance(prefix_val: float, active_val: float) -> float:
            return max(
                underflow_tol,
                1e-7 * max(1.0, abs(prefix_val), abs(active_val)),
            )

        # Require close agreement between cached and prefix-sum views.
        tolerance = _mismatch_tolerance(prefix_L, pool.L_active)
        if abs(prefix_L - pool.L_active) > tolerance:
            try:
                t_now = t
            except NameError:
                t_now = None
            t_clause = f" t={t_now}" if t_now is not None else ""
            raise AssertionError(
                f"L_active mismatch ({label}){t_clause} tick={pool.tick} active={pool.L_active} prefix={prefix_L}"
            )

        # If we have meaningful active liquidity, ensure price lies inside the band.
        if pool.L_active > EPS_LIQ:
            sa = pool.s_lower()
            sb = pool.s_upper()
            band_scale = max(1.0, abs(sa), abs(sb), abs(pool.S))
            boundary_tol = EPS_BOUNDARY * band_scale
            if pool.S < sa - boundary_tol or pool.S > sb + boundary_tol:
                raise AssertionError(
                    f"Price S={pool.S} outside active band during {label}: tick={pool.tick} band=[{sa},{sb}]"
                )
            # Snap tiny floating-point drift so pool.S stays on the boundary.
            if pool.S < sa:
                pool.S = sa
            elif pool.S > sb:
                pool.S = sb

    for lp in LPs:
        for pos in lp.positions:
            _register_position(pos)

    # ------------------ Recorders ------------------
    P_series, M_series = [], []
    X_active_end, Y_active_end = [], []
    # No-arb band constructed from the validated snapshot CEX price at the beginning
    # of the block (equal to the end of the previous block).
    band_lo_target, band_hi_target = [], []
    L_end, L_pre_step = [], []
    L_pre_trader, L_pre_arb_eff = [], []
    trader_y_series, arb_y_series = [], []
    trader_steps, trader_dirs = [], []
    arb_steps, arb_dirs = [], []
    mint_steps, mint_sizes, burn_steps, burn_sizes = [], [], [], []
    mint_is_passive: List[bool] = []
    burn_is_passive: List[bool] = []
    mint_is_jiter: List[bool] = []
    burn_is_jiter: List[bool] = []
    jiter_activity_steps: List[int] = []
    jiter_activity_signs: List[int] = []
    # --- Agent activity recorders (+1 / -1 per action) ---
    smart_activity_steps: List[int] = []
    smart_activity_signs: List[int] = []
    noise_activity_steps: List[int] = []
    noise_activity_signs: List[int] = []
    arb_skip_steps: List[int] = []
    mint_widths = []
    w_ticks_series: List[int] = []
    w_unclipped_series: List[float] = []
    w_noise_series: List[float] = []
    liq_history: List[Dict[int, float]] = []
    tick_history: List[int] = []
    delta_a_cex_series = []
    cex_sigma_series: List[float] = []
    cex_regime_series: List[str] = []
    # --- Block-start no-arb band (validated snapshot CEX price) ---
    # (Stored as band_lo_target / band_hi_target for backward naming in internals.)
    # --- Micro-time traces (mempool micro-steps) ---
    micro_steps, M_micro, P_micro = [], [], []
    micro_valid_steps, micro_valid_prices = [], []
    micro_counter = 0
    # --- PnL recorders ---
    trader_pnl_steps = []       # realized per-step PnL 
    arb_pnl_steps = []          # realized per-step PnL 
    lp_pnl_total_series = []    # cumulative hedged PnL (fees - rebal) across all LPs
    lp_pnl_active_series = []   # cumulative hedged PnL for active (narrow) LPs
    lp_pnl_passive_series = []  # cumulative hedged PnL for passive LPs
    lp_unhedged_total_series = []   # V^{LP}_t - V^{LP}_0 across all LPs
    lp_unhedged_active_series = []
    lp_unhedged_passive_series = []
    lp_rebal_total_series = []  # cumulative rebalancing PnL (benchmark) across LPs
    lp_rebal_active_series = []
    lp_rebal_passive_series = []
    lp_rebal_value_total_series = []   # V^{reb}_t
    lp_rebal_value_active_series = []
    lp_rebal_value_passive_series = []
    lp_fee_value_total_series = []     # F_t (cumulative fees, marked to m_t)
    lp_fee_value_active_series = []
    lp_fee_value_passive_series = []
    # Cumulative fee counters in token units (no mark-to-market).
    # These are useful for constructing *per-block fee flow value*:
    #   ΔF_flow,t = (Δfees0_earned,t) * m_t + (Δfees1_earned,t)
    # which avoids revaluing previously earned token0 fees when m moves.
    lp_fees0_earned_total_series: List[float] = []
    lp_fees1_earned_total_series: List[float] = []
    lp_fees0_earned_active_series: List[float] = []
    lp_fees1_earned_active_series: List[float] = []
    lp_fees0_earned_passive_series: List[float] = []
    lp_fees1_earned_passive_series: List[float] = []
    lp_lvr_total_series = []           # F_t - hedged = LVR_t
    lp_lvr_active_series = []
    lp_lvr_passive_series = []
    jiter_wallet_series: List[float] = []
    jiter_wealth_series: List[float] = []
    jiter_fee_value_series: List[float] = []
    jiter_fees0_earned_series: List[float] = []
    jiter_fees1_earned_series: List[float] = []
    jiter_position_value_series: List[float] = []
    jiter_pnl_series: List[float] = []
    jiter_flash_fee_paid_series: List[float] = []
    trader_exec_count = []
    arb_exec_count = []

    # --- Split PnL/flow recorders for Smart Router vs Noise Trader ---
    sr_pnl_steps = []
    noise_pnl_steps = []
    sr_exec_count = []
    noise_exec_count = []
    sr_y_series = []
    noise_y_series = []
    dex_notional_y_series = []
    sr_cex_exec_count = []
    sr_dex_exec_count = []

    pid_str = str(os.getpid())
    LOG_BUFFER_LIMIT = 10_000
    log_buffer: List[str] = []
    verbose_log_path: Optional[Path] = None
    verbose_log_path_str = ""
    verbose_log = None

    if light_mode:
        def buffer_log(msg: str) -> None:
            return None

        def flush_log_buffer() -> None:
            return None
    else:
        # Determine verbose log file path for this run (scenario-aware)
        logs_dir = results_root_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        verbose_log_path = next_numbered_path(logs_dir / f"{pid_str}_verbose_steps_{fee_mode}")
        verbose_log_path_str = str(verbose_log_path)

        verbose_log = open(verbose_log_path_str, "a")

        def buffer_log(msg: str) -> None:
            """Accumulate log lines before flushing to disk."""
            log_buffer.append(msg)
            if len(log_buffer) >= LOG_BUFFER_LIMIT:
                verbose_log.write("".join(log_buffer))
                log_buffer.clear()

        def flush_log_buffer() -> None:
            """Write the buffered log entries to disk."""
            if log_buffer:
                verbose_log.write("".join(log_buffer))
                log_buffer.clear()

        run_timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        buffer_log(f"# PID {pid_str}\n")
        buffer_log(f"# Run date {run_timestamp}\n")
        buffer_log("# Simulation parameters\n")
        for key in sorted(initial_params):
            buffer_log(f"{key} = {initial_params[key]}\n")
        buffer_log("\n")


    # --- LP wealth recorders (new) ---
    lp_wallet_series = []      # realized wallet 
    lp_wealth_series = []      # wallet + open PnL 
    lp_wallet_active_series = []   # active narrow LPs
    lp_wallet_passive_series = []  # passive LPs
    lp_wealth_active_series = []
    lp_wealth_passive_series = []
    # --- Dynamic fee signal recorders (new) ---
    fee_sigma_series = []          # EWMA abs log-return (σ̂)
    fee_basis_ticks_series = []    # EWMA fee-adjusted basis, in ticks
    fee_imb_series = []            # EWMA |imbalance| in [0,1]
    fee_signal_series = []         # controller signal actually used (per fee_mode)

    if light_mode:
        P_series = _NullList()
        M_series = _NullList()
        X_active_end = _NullList()
        Y_active_end = _NullList()
        # (Only keep the snapshot-based no-arb band; pre/post variants removed.)
        L_end = _NullList()
        L_pre_step = _NullList()
        L_pre_trader = _NullList()
        L_pre_arb_eff = _NullList()
        trader_y_series = _NullList()
        arb_y_series = _NullList()
        trader_steps = _NullList()
        trader_dirs = _NullList()
        arb_steps = _NullList()
        arb_dirs = _NullList()
        mint_sizes = _NullList()
        burn_sizes = _NullList()
        mint_is_passive = _NullList()
        burn_is_passive = _NullList()
        mint_is_jiter = _NullList()
        burn_is_jiter = _NullList()
        jiter_activity_steps = _NullList()
        jiter_activity_signs = _NullList()
        smart_activity_steps = _NullList()
        smart_activity_signs = _NullList()
        noise_activity_steps = _NullList()
        noise_activity_signs = _NullList()
        arb_skip_steps = _NullList()
        mint_widths = _NullList()
        w_ticks_series = _NullList()
        w_unclipped_series = _NullList()
        w_noise_series = _NullList()
        liq_history = _NullList()
        tick_history = _NullList()
        delta_a_cex_series = _NullList()
        cex_sigma_series = _NullList()
        cex_regime_series = _NullList()
        band_lo_target = _NullList()
        band_hi_target = _NullList()
        micro_steps = _NullList()
        M_micro = _NullList()
        P_micro = _NullList()
        micro_valid_steps = _NullList()
        micro_valid_prices = _NullList()
        trader_pnl_steps = _NullList()
        lp_pnl_total_series = _NullList()
        lp_unhedged_total_series = _NullList()
        lp_rebal_total_series = _NullList()
        lp_rebal_active_series = _NullList()
        lp_rebal_passive_series = _NullList()
        lp_rebal_value_total_series = _NullList()
        lp_rebal_value_active_series = _NullList()
        lp_rebal_value_passive_series = _NullList()
        lp_fee_value_total_series = _NullList()
        lp_fee_value_active_series = _NullList()
        lp_fee_value_passive_series = _NullList()
        lp_fees0_earned_total_series = _NullList()
        lp_fees1_earned_total_series = _NullList()
        lp_fees0_earned_active_series = _NullList()
        lp_fees1_earned_active_series = _NullList()
        lp_fees0_earned_passive_series = _NullList()
        lp_fees1_earned_passive_series = _NullList()
        lp_lvr_total_series = _NullList()
        lp_lvr_active_series = _NullList()
        lp_lvr_passive_series = _NullList()
        jiter_wallet_series = _NullList()
        jiter_wealth_series = _NullList()
        jiter_fee_value_series = _NullList()
        jiter_fees0_earned_series = _NullList()
        jiter_fees1_earned_series = _NullList()
        jiter_position_value_series = _NullList()
        jiter_pnl_series = _NullList()
        jiter_flash_fee_paid_series = _NullList()
        trader_exec_count = _NullList()
        arb_exec_count = _NullList()
        sr_exec_count = _NullList()
        noise_exec_count = _NullList()
        sr_y_series = _NullList()
        noise_y_series = _NullList()
        dex_notional_y_series = _NullList()
        # Keep smart-router exec counts even in light_mode for downstream DEX-share aggregation.
        sr_cex_exec_count = []
        sr_dex_exec_count = []
        lp_wallet_series = _NullList()
        lp_wallet_active_series = _NullList()
        lp_wallet_passive_series = _NullList()
        lp_wealth_series = _NullList()
        lp_wealth_active_series = _NullList()
        lp_wealth_passive_series = _NullList()
        fee_sigma_series = _NullList()
        fee_basis_ticks_series = _NullList()
        fee_imb_series = _NullList()
        fee_signal_series = _NullList()
    # --- EWMA volatility signal for active LP width rule ---
    # Paper spec: width depends on an EWMA of a volatility-related signal. We use
    # EWMA(|log-return of the CEX mid|) with half-life `basis_half_life` (config name kept
    # for backward compatibility with existing scenario files).
    ewma_width = EWMA(half_life_steps=basis_half_life)
    prev_m_for_width = ref.m

    # --- Dynamic fee controller state (new) ---
    pool.f = float(f0)  # initial fee overrides builder default
    fee_next: Optional[float] = None
    fee_cooldown_left: int = 0
    fee_series: List[float] = []

    # EWMA signals for controllers
    ewma_sigma_fee = EWMA(half_life_steps=fee_half_life, init=0.0)  # |log m_t - log m_{t-1}|
    ewma_basis_fee = EWMA(half_life_steps=fee_half_life, init=0.0)  # fee-adjusted log gap
    # EWMA of (dLVR - dFees) normalized by DEX notional 
    ewma_lvr_gap_fee = EWMA(half_life_steps=fee_half_life, init=0.0)
    prev_cex_for_vol = ref.m   # for volatility_cex mode
    prev_dex_for_vol = pool.price  # for volatility_dex mode
    prev_lp_fee_value_total = 0.0
    prev_lp_lvr_total = 0.0

    # ------------------ LVR rebalancer helpers ------------------
    REBAL_EPS = 1e-18
    _last_rebalance_S: float = -1.0  # Track last S where we did full rebalance

    def _ensure_rebalancer_initialized(lp: LPAgent, M_now: float, S_now: float) -> None:
        rb = lp.rebalancer
        if rb.initialized:
            return
        rb.reset()
        x_target = lp_token0_exposure(lp, S_now)
        rb.x_prev = x_target
        rb.cumulative_R = 0.0
        rb.last_M = M_now
        wealth_now = lp_wealth_y(lp, S_now, M_now)
        rb.initial_lp_value_y = wealth_now
        rb.initial_rebal_value_y = wealth_now  # V^{reb}_0 = V^{LP}_0
        # Maintain a coherent benchmark balance sheet for debugging/diagnostics:
        # V^{reb}_0 = x_0 * M_0 + cash_0.
        rb.cash_y = rb.initial_rebal_value_y - rb.x_prev * M_now
        rb.last_wealth_y = wealth_now
        rb.last_cumulative_R = 0.0
        rb.hedged_pnl_cum = 0.0
        rb.initialized = True

    def _rebalance_lp_to_target(lp: LPAgent, M_now: float, S_now: float) -> None:
        _ensure_rebalancer_initialized(lp, M_now, S_now)
        rb = lp.rebalancer
        x_target = lp_token0_exposure(lp, S_now)
        dx = x_target - rb.x_prev
        if abs(dx) > REBAL_EPS:
            rb.cash_y -= dx * M_now
            rb.x_prev = x_target
        rb.last_M = M_now

    def _rebalance_by_ids(lp_ids: Set[int], M_now: float, S_now: float) -> None:
        if not lp_ids:
            return
        for lp_id in lp_ids:
            lp = lp_lookup.get(lp_id)
            if lp is not None:
                _rebalance_lp_to_target(lp, M_now, S_now)

    def _rebalance_all(M_now: float, S_now: float) -> None:
        nonlocal _last_rebalance_S
        # Skip full rebalance if S hasn't changed (exposure is same for all LPs)
        # This is safe because position changes trigger explicit rebalance via _rebalance_lp_to_target
        if S_now == _last_rebalance_S:
            # Still need to update last_M for CEX price tracking
            for lp in LPs:
                rb = lp.rebalancer
                if rb.initialized:
                    rb.last_M = M_now
            return
        _last_rebalance_S = S_now
        for lp in LPs:
            _rebalance_lp_to_target(lp, M_now, S_now)

    def _accrue_price_move(lp: LPAgent, M_new: float) -> None:
        rb = lp.rebalancer
        if not rb.initialized:
            rb.last_M = M_new
            return
        delta = M_new - rb.last_M
        if abs(delta) > 0.0:
            rb.cumulative_R += rb.x_prev * delta
            rb.last_M = M_new

    def _broadcast_price_move(M_new: float) -> None:
        for lp in LPs:
            _accrue_price_move(lp, M_new)

    # Initialize rebalancers to match current exposures before the simulation loop
    _rebalance_all(ref.m, pool.S)

    # ------------------ Helpers ------------------
    # Batched rebalancing: collect touched LP IDs during swap, flush after swap completes
    _pending_rebalance_ids: Set[int] = set()

    def _flush_pending_rebalance() -> None:
        """Rebalance all LPs touched during a swap, then clear the set."""
        nonlocal _pending_rebalance_ids
        if _pending_rebalance_ids:
            # Take a snapshot to avoid modification during iteration
            ids_to_rebalance = set(_pending_rebalance_ids)
            _pending_rebalance_ids = set()  # Clear before processing to avoid race conditions
            _rebalance_by_ids(ids_to_rebalance, ref.m, pool.S)

    def allocate_fees(token: str, fee_amt: float, tick_snapshot: int, L_snapshot: float) -> None:
        """Allocate fees to positions and mark LPs for rebalancing.
        
        Note: This function is called by the pool swap methods. Even if fee_amt is zero,
        we still need to track touched positions for rebalancing to keep hedged PnL / LVR
        bookkeeping correct.
        """
        bucket = positions_by_tick.get(tick_snapshot)
        if not bucket:
            return
        
        # Always track touched LP IDs for rebalancing, even if no fees to distribute.
        # This ensures rebalancer exposure is updated when price moves, regardless of fee level.
        for pos in bucket:
            if pos.L > 0.0:
                _pending_rebalance_ids.add(pos.owner)
        
        # If no fees to distribute, we're done (but LPs are still marked for rebalancing)
        if fee_amt <= 0:
            return
        
        total_L = 0.0
        for pos in bucket:
            L_pos = pos.L
            if L_pos > 0.0:
                total_L += L_pos
        if total_L <= 0.0:
            return

        inv_total_L = 1.0 / total_L
        owner_fee: Dict[int, float] = {}

        if token == "x":
            for pos in bucket:
                L_pos = pos.L
                if L_pos <= 0.0:
                    continue
                fee_for_pos = fee_amt * L_pos * inv_total_L
                if fee_for_pos == 0.0:
                    continue
                pos.fees0 += fee_for_pos
                owner_id = pos.owner
                owner_fee[owner_id] = owner_fee.get(owner_id, 0.0) + fee_for_pos

            for owner_id, fee in owner_fee.items():
                if fee == 0.0:
                    continue
                lp = lp_lookup.get(owner_id)
                if lp is not None:
                    lp.fees0_earned = getattr(lp, "fees0_earned", 0.0) + fee
                # Note: LP is already marked for rebalancing at the start of allocate_fees
        else:
            for pos in bucket:
                L_pos = pos.L
                if L_pos <= 0.0:
                    continue
                fee_for_pos = fee_amt * L_pos * inv_total_L
                if fee_for_pos == 0.0:
                    continue
                pos.fees1 += fee_for_pos
                owner_id = pos.owner
                owner_fee[owner_id] = owner_fee.get(owner_id, 0.0) + fee_for_pos

            for owner_id, fee in owner_fee.items():
                if fee == 0.0:
                    continue
                lp = lp_lookup.get(owner_id)
                if lp is not None:
                    lp.fees1_earned = getattr(lp, "fees1_earned", 0.0) + fee
                # Note: LP is already marked for rebalancing at the start of allocate_fees

    def burn_any(lp: LPAgent, idx: int, *, m_ref: float) -> None:
        nonlocal delta_a_cex_this
        pos = lp.positions.pop(idx)
        # Cash-budgeted LP accounting: burning returns underlying tokens + fees,
        # then the LP immediately converts any token0 into token1 on the CEX.
        amt0, amt1 = pos.current_amounts(pool.S)
        amt0_total = float(amt0) + float(pos.fees0)
        amt1_total = float(amt1) + float(pos.fees1)
        lp.wallet_x = 0.0
        lp.wallet_y = float(getattr(lp, "wallet_y", 0.0)) + amt1_total + amt0_total * float(m_ref)
        # Selling token0 on the CEX contributes negative Δa (immediate impact)
        if abs(amt0_total) > 1e-18:
            delta_a_cex_this += -amt0_total
            ref.apply_impact_only(-amt0_total)
            _broadcast_price_move(ref.m)
        _unregister_position(pos)
        pool.add_liquidity_range(pos.lower, pos.upper, -pos.L)
        _assert_active_liquidity_state_fast("lp_burn")

        burn_steps.append(t)
        burn_sizes.append(pos.L)
        burn_is_passive.append(bool(lp.is_passive))
        burn_is_jiter.append(False)

        buffer_log(
            f"[t={t:03d}] LP{lp.id} BURN L={pos.L:.4f} [{pos.lower},{pos.upper}) | "
            f"L_active={pool.L_active:.4f} | tick={pool.tick} (impact applied)\n"
        )

        lp.cooldown = np.random.randint(3, 9)  # 3–8 steps of "hands off"
        _rebalance_lp_to_target(lp, ref.m, pool.S)


    def reserves_in_active_tick() -> Tuple[float, float]:
        if pool.L_active <= EPS_LIQ:
            return 0.0, 0.0
        sa, sb = pool.s_lower(), pool.s_upper()
        S_eff = min(max(pool.S, sa), sb)
        x = pool.L_active * max(0.0, 1.0 / S_eff - 1.0 / sb)
        y = pool.L_active * max(0.0, S_eff - sa)
        return x, y

    # -------------------------------------------------------------------------
    # Cash-budgeted LP mint helper (LP wallet in token1 only)
    # -------------------------------------------------------------------------
    def _draw_wallet_utilization_factor() -> float:
        """
        Draw η in (0, 1] controlling wallet utilization for a mint attempt.

        Paper spec: Z ~ LogNormal(mint_mu, mint_sigma), η = min(1, Z).
        """
        z = float(np.random.lognormal(mint_mu, mint_sigma))
        if not math.isfinite(z) or z <= 0.0:
            return 0.0
        return min(1.0, z)
    
    # def _draw_wallet_utilization_factor() -> float:
    #     """
    #     Draw η in (0, 1] controlling wallet utilization for a mint attempt.

    #     Paper spec: Z ~ LogNormal(mint_mu, mint_sigma), η = min(1, Z).
    #     """
    #     z = float(np.random.lognormal(mint_mu, mint_sigma))
    #     return z / (1.0 + z)

    def _max_feasible_liquidity_from_cash(
        *,
        cash_y: float,
        sa: float,
        sb: float,
        S_ref: float,
        m_ref: float,
    ) -> float:
        """
        Compute L^max for a proposed range [sa, sb) given a cash-only budget in token1.

        Uses deposit coefficients implied by Uniswap v3 mint math at the chosen
        reference state (S_ref; sa, sb):

            (Δx0, Δx1) = (a0 L, a1 L),

        so the required token1 value to mint liquidity L is:

            cost_y(L) = Δx0(L) * m_ref + Δx1(L) = (a0 * m_ref + a1) * L.
        """
        a0, a1 = minted_amounts_at_S(1.0, sa, sb, S_ref)
        denom = float(a0) * float(m_ref) + float(a1)
        if denom <= 0.0:
            return 0.0
        L_max = float(cash_y) / denom
        if not math.isfinite(L_max) or L_max <= 0.0:
            return 0.0
        return float(L_max)

    def _execute_cash_budgeted_mint(
        *,
        lp: LPAgent,
        lower: int,
        upper: int,
        eta: float,
        S_ref: float,
        m_ref: float,
        log_prefix: str,
        is_jiter: bool,
    ) -> Optional[Position]:
        """
        Execute a cash-budgeted mint for a (possibly pre-chosen) range.

        The LP holds token1 cash only. At mint time, they conceptually purchase the
        required token0 amount on the CEX at price m_ref (impacting the CEX through
        `delta_a_cex_this`), and then deposit the (token0, token1) amounts into the AMM.

        Returns the created Position on success; otherwise returns None.
        """
        nonlocal delta_a_cex_this
        if upper <= lower:
            return None

        eta_f = float(eta)
        if not math.isfinite(eta_f) or eta_f <= 0.0:
            return None
        eta_f = min(1.0, eta_f)

        sa, sb = pool.s_lower(lower), pool.s_lower(upper)

        cash_y = float(getattr(lp, "wallet_y", 0.0))
        if not is_jiter:
            cash_y = max(0.0, cash_y)

        L_max = _max_feasible_liquidity_from_cash(
            cash_y=cash_y,
            sa=sa,
            sb=sb,
            S_ref=S_ref,
            m_ref=m_ref,
        )
        if L_max <= 0.0:
            return None

        L_new = eta_f * L_max
        if L_new <= 0.0:
            return None

        amt0, amt1 = minted_amounts_at_S(L_new, sa, sb, S_ref)
        amt0 = float(amt0)
        amt1 = float(amt1)
        if not (math.isfinite(amt0) and math.isfinite(amt1)):
            return None

        cost_y = amt0 * float(m_ref) + amt1

        # Enforce feasibility for non-JIT LPs (allow JIT to go negative as "flash-funded").
        eps = 1e-12
        if not is_jiter:
            if cost_y > cash_y + eps:
                return None
        lp.wallet_x = 0.0
        lp.wallet_y = float(getattr(lp, "wallet_y", 0.0)) - cost_y
        # Buying token0 on the CEX contributes positive Δa (immediate impact)
        if abs(amt0) > 1e-18:
            delta_a_cex_this += +amt0
            ref.apply_impact_only(+amt0)
            _broadcast_price_move(ref.m)

        pos = Position(
            owner=lp.id,
            lower=lower,
            upper=upper,
            L=L_new,
            sa=sa,
            sb=sb,
            amt0_init=amt0,
            amt1_init=amt1,
            hodl0_value_y=amt0 * float(m_ref) + amt1,
        )
        pool.add_liquidity_range(lower, upper, L_new)
        lp.positions.append(pos)
        _register_position(pos)
        _assert_active_liquidity_state_fast(f"{log_prefix}_mempool")

        mint_steps.append(t)
        mint_sizes.append(L_new)
        mint_widths.append(upper - lower)
        mint_is_passive.append(bool(getattr(lp, "is_passive", False)))
        mint_is_jiter.append(bool(is_jiter))

        buffer_log(
            f"[t={t:03d}] {log_prefix} L={L_new:.4f} [{lower},{upper}) | "
            f"L_active={pool.L_active:.4f} | tick={pool.tick}\n"
        )
        _rebalance_lp_to_target(lp, ref.m, pool.S)
        return pos

    # ----- Arbitrage internals  -----
    def fast_span_up(to_S: float, target_S: float) -> Tuple[float, float, float]:
        S0, L, r = pool.S, pool.L_active, pool.r
        S1 = min(to_S, target_S)
        if S1 <= S0 or L <= EPS_LIQ:
            return 0.0, 0.0, 0.0
        dy_eff = L * (S1 - S0)
        dy_pre = dy_eff / r
        dx_out = L * (1 / S0 - 1 / S1)
        fee_y = dy_pre - dy_eff
        pool.S = S1
        return dy_pre, dx_out, fee_y

    def fast_span_down(to_S: float, target_S: float) -> Tuple[float, float, float]:
        S0, L, r = pool.S, pool.L_active, pool.r
        S1 = max(to_S, target_S)
        if S1 >= S0 or L <= EPS_LIQ:
            return 0.0, 0.0, 0.0
        dx_eff = L * (1 / S1 - 1 / S0)
        dx_pre = dx_eff / r
        dy_out = L * (S0 - S1)
        fee_x = dx_pre - dx_eff
        pool.S = S1
        return dx_pre, dy_out, fee_x

    def swap_exact_to_target(target_price: float, direction: str, fee_cb: Optional[Callable[[str, float, int, float], None]] = None) -> Tuple[float, float, float]:
        target_S = math.sqrt(max(1e-18, target_price))

        total_in = total_out = 0.0
        L_first = 0.0
        if pool.L_active <= EPS_LIQ:
            return 0.0, 0.0, 0.0

        if direction == "up":
            while pool.L_active > EPS_LIQ and pool.S < target_S - EPS_BOUNDARY:
                S_hi = pool.s_upper()
                L_before = pool.L_active
                _tick_before = pool.tick
                dy, dx, f = fast_span_up(S_hi, target_S)
                if dy > 0 and L_first == 0.0:
                    L_first = L_before
                # Per-span fee allocation (token Y)
                if fee_cb and f > 0.0 and L_before > 0.0:
                    fee_cb("y", f, _tick_before, L_before)
                total_in += dy
                total_out += dx
                if pool.S >= target_S - EPS_BOUNDARY:
                    break
                pool._cross_up_once()
                _assert_active_liquidity_state_fast("swap_exact_to_target:cross_up")
                if pool.L_active <= EPS_LIQ:
                    break
            _assert_active_liquidity_state_fast("swap_exact_to_target:up_end")
            return total_in, total_out, L_first

        else:  # "down"
            while pool.L_active > EPS_LIQ and pool.S > target_S + EPS_BOUNDARY:
                S_lo = pool.s_lower()
                L_before = pool.L_active
                _tick_before = pool.tick
                dx, dy, f = fast_span_down(S_lo, target_S)
                if dx > 0 and L_first == 0.0:
                    L_first = L_before
                # Per-span fee allocation (token X)
                if fee_cb and f > 0.0 and L_before > 0.0:
                    fee_cb("x", f, _tick_before, L_before)
                total_in += dx
                total_out += dy
                if pool.S <= target_S + EPS_BOUNDARY:
                    break
                pool._cross_down_once()
                _assert_active_liquidity_state_fast("swap_exact_to_target:cross_down")
                if pool.L_active <= EPS_LIQ:
                    break
            _assert_active_liquidity_state_fast("swap_exact_to_target:down_end")
            return total_in, total_out, L_first


    def arbitrage_to_target(arb_ref_m: float) -> Tuple[float, float, float, Optional[str], float]:
        """
        Returns:
        in_used        = total input amount into the DEX (dy for 'up', dx for 'down')
        x_out_from_dex = token A out from the DEX (dx_out for 'up'; 0.0 for 'down')
        y_out_from_dex = token B out from the DEX (0.0 for 'up'; dy_out for 'down')
        direction      = 'up' or 'down' or None
        L_first
        """
        P = pool.price
        r = pool.r
        lo = (arb_ref_m * r) / flash_loan_mult
        hi = (arb_ref_m * flash_loan_mult) / max(r, 1e-18)
        if P < lo * (1 - 1e-9):
            # up: returns (dy_in, dx_out, 0.0, direction, L_first)
            dy_in, dx_out, Lff = swap_exact_to_target(lo, "up", fee_cb=allocate_fees)
            _flush_pending_rebalance()  # Batch rebalance all LPs touched during arb
            return dy_in, dx_out, 0.0, ("up" if dy_in > 0 else None), Lff
        if P > hi * (1 + 1e-9):
            # down: returns (dx_in, 0.0, dy_out, direction, L_first)
            dx_in, dy_out, Lff = swap_exact_to_target(hi, "down", fee_cb=allocate_fees)
            _flush_pending_rebalance()  # Batch rebalance all LPs touched during arb
            return dx_in, 0.0, dy_out, ("down" if dx_in > 0 else None), Lff
        return 0.0, 0.0, 0.0, None, 0.0

    def _clone_pool_for_preview() -> V3Pool:
        """Shallow pool clone to dry-run arbitrage profitability without touching state."""
        return V3Pool(
            g=pool.g,
            base_s=pool.base_s,
            tick=pool.tick,
            S=pool.S,
            f=pool.f,
            # liquidity_net=dict(pool.liquidity_net),
            liquidity_net=pool.liquidity_net,
            tick_spacing=pool.tick_spacing,
        )

    def _preview_swap_exact_to_target(pool_obj: V3Pool, target_price: float, direction: str) -> Tuple[float, float, float]:
        """
        Dry-run variant of swap_exact_to_target used to estimate arb profitability.
        Mutates the provided pool_obj clone; does NOT allocate fees or touch real state.
        """
        target_S = math.sqrt(max(1e-18, target_price))

        total_in = total_out = 0.0
        L_first = 0.0

        if pool_obj.L_active <= EPS_LIQ:
            return 0.0, 0.0, 0.0

        if direction == "up":
            while pool_obj.L_active > EPS_LIQ and pool_obj.S < target_S - EPS_BOUNDARY:
                S_hi = pool_obj.s_upper()
                if pool_obj.S >= S_hi - EPS_BOUNDARY:
                    pool_obj.S = S_hi
                    pool_obj._cross_up_once()
                    continue
                L_before = pool_obj.L_active
                dy_eff = L_before * (min(S_hi, target_S) - pool_obj.S)
                if dy_eff <= 0.0:
                    break
                dy_pre = dy_eff / pool_obj.r
                dx_out = L_before * (1 / pool_obj.S - 1 / (pool_obj.S + dy_eff / L_before))
                pool_obj.S = min(S_hi, target_S)

                if dy_pre > 0 and L_first == 0.0:
                    L_first = L_before
                total_in += dy_pre
                total_out += dx_out

                if pool_obj.S >= target_S - EPS_BOUNDARY:
                    return total_in, total_out, L_first
                pool_obj._cross_up_once()

        else:
            while pool_obj.L_active > EPS_LIQ and pool_obj.S > target_S + EPS_BOUNDARY:
                S_lo = pool_obj.s_lower()
                if pool_obj.S <= S_lo + EPS_BOUNDARY:
                    pool_obj.S = S_lo
                    pool_obj._cross_down_once()
                    continue
                L_before = pool_obj.L_active
                dx_eff = L_before * (1 / max(target_S, S_lo) - 1 / pool_obj.S)
                if dx_eff <= 0.0:
                    break
                dx_pre = dx_eff / pool_obj.r
                dy_out = L_before * (pool_obj.S - max(target_S, S_lo))
                pool_obj.S = max(target_S, S_lo)

                if dx_pre > 0 and L_first == 0.0:
                    L_first = L_before
                total_in += dx_pre
                total_out += dy_out

                if pool_obj.S <= target_S + EPS_BOUNDARY:
                    return total_in, total_out, L_first
                pool_obj._cross_down_once()

        return total_in, total_out, L_first

    def preview_arbitrage_to_target(arb_ref_m: float) -> Tuple[float, float, float, Optional[str], float]:
        """
        Dry-run arbitrage path on a cloned pool to estimate profitability before executing.
        Returns the same tuple shape as arbitrage_to_target but without mutating real state.
        """
        pool_sim = _clone_pool_for_preview()
        P = pool_sim.price
        r = pool_sim.r
        lo = (arb_ref_m * r) / flash_loan_mult
        hi = (arb_ref_m * flash_loan_mult) / max(r, 1e-18)
        if P < lo * (1 - 1e-9):
            dy_in, dx_out, Lff = _preview_swap_exact_to_target(pool_sim, lo, "up")
            return dy_in, dx_out, 0.0, ("up" if dy_in > 0 else None), Lff
        if P > hi * (1 + 1e-9):
            dx_in, dy_out, Lff = _preview_swap_exact_to_target(pool_sim, hi, "down")
            return dx_in, 0.0, dy_out, ("down" if dx_in > 0 else None), Lff
        return 0.0, 0.0, 0.0, None, 0.0

    # Per-micro-step Poisson intensities
    smart_lambda_micro_step = smart_lambda_micro
    noise_lambda_micro_step = noise_lambda_micro

    total_noise_swaps_executed = 0
    total_noise_swaps_skipped = 0
    total_smart_swaps_executed = 0
    total_smart_swaps_skipped = 0
    total_arb_swaps_executed = 0
    total_arb_no_op_in_band = 0
    total_arb_swaps_rejected_profitability = 0
    total_jit_trades_executed = 0
    smart_swaps_x_to_y = 0
    smart_swaps_y_to_x = 0
    noise_swaps_x_to_y = 0
    noise_swaps_y_to_x = 0
    # Track the last validated DEX state (end of previous block)
    validated_S = pool.S
    validated_tick = pool.tick
    validated_cex = ref.m
    agent_S_ref = validated_S
    agent_tick_ref = validated_tick
    cex_ref_for_agents = validated_cex
    def _baseline_quote_x_to_y(dx: float) -> float:
        """Reference quote using last validated price (pre-block) and current fee."""
        if dx <= 0.0:
            return 0.0
        price_ref = agent_S_ref * agent_S_ref
        return dx * pool.r * price_ref

    def _baseline_quote_y_to_x(dy: float) -> float:
        """Reference quote using last validated price (pre-block) and current fee."""
        if dy <= 0.0:
            return 0.0
        price_ref = agent_S_ref * agent_S_ref
        return (dy * pool.r) / max(price_ref, 1e-18)

    def _rollback_pool_state(tick_before: int, S_before: float) -> None:
        """Restore pool tick/S and derived active liquidity after a no-op swap."""
        pool.tick = tick_before
        pool.recompute_active_L()
        pool.S = S_before

    def _draw_trader_notional() -> float:
        """Sample a token1 notional for trader orders (shared across directions)."""
        return float(np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma)))

    # ------------------ Main loop ------------------
    # tqdm is convenient for interactive runs but adds overhead and emits output
    # even when `verbose=False`; disable it in that case.
    for t in tqdm(range(T), desc="Simulating ABM", unit=" step", disable=not verbose):
        agent_S_ref = validated_S
        agent_tick_ref = validated_tick
        cex_ref_for_agents = validated_cex
        # --- Apply any committed fee update (commit→reveal) ---
        if fee_cooldown_left > 0:
            fee_cooldown_left -= 1
        if fee_next is not None and fee_cooldown_left <= 0:
            pool.f = clamp(fee_next, f_min, f_max)
            fee_next = None
        r = pool.r

        # Start-of-step rebalance benchmark update (predictable integrand)
        _rebalance_all(ref.m, pool.S)

        # Record start-of-step active L and price
        L_pre_step.append(pool.L_active)
        P_before = pool.price

        # ---------------------------------------------------------------------
        # Active LP width rule: EWMA of CEX volatility + binomial noise
        # ---------------------------------------------------------------------
        # Use an EWMA of absolute CEX log-returns as the volatility signal.
        try:
            log_m_now = math.log(max(ref.m, 1e-18))
            log_m_prev = math.log(max(prev_m_for_width, 1e-18))
            vol_obs = abs(log_m_now - log_m_prev)
        except ValueError:
            vol_obs = 0.0
        prev_m_for_width = ref.m
        vol_hat = ewma_width.update(vol_obs)
        vol_in_ticks = vol_hat / TICK_LN

        # --- Mean-zero binomial noise term (in ticks) ---
        # draw K ~ Bin(n, p), center by n p, and scale by tick_spacing to live on the grid
        noise_ticks = 0.0
        if binom_n > 0 and 0.0 < binom_p < 1.0:
            K = np.random.binomial(binom_n, binom_p)
            noise_ticks = (K - binom_n * binom_p) * pool.tick_spacing  # non-negative noise per spec

        # Map to width in ticks: w = clip(w_min + slope * vol_in_ticks + noise_ticks, w_min, w_max)
        w_unclipped = w_min_ticks + slope_s * vol_in_ticks + noise_ticks
        step_width_ticks = pool.tick_spacing  # total width snaps to tick_spacing
        w_ticks = int(round(w_unclipped / step_width_ticks)) * step_width_ticks
        # Enforce minimum based on w_min_ticks (rounded up to spacing multiple), not just one band
        _min_bands = max(1, (w_min_ticks + step_width_ticks - 1) // step_width_ticks)
        _max_bands = max(1, w_max_ticks // step_width_ticks)
        w_ticks = max(_min_bands * step_width_ticks, min(w_ticks, _max_bands * step_width_ticks))
        w_ticks_series.append(w_ticks)
        w_unclipped_series.append(w_unclipped)
        w_noise_series.append(noise_ticks)
        # ---------------------------------------------------------------------

        # --- Per-step accumulators (so we can randomize actor order) ---
        trader_y_this = 0.0
        arb_y_this = 0.0
        dex_notional_y_this = 0.0
        trader_pnl_this = 0.0
        arb_pnl_this = 0.0
        _trader_execs = 0
        _arb_execs = 0
        # Split per-actor accumulators
        sr_acc = TraderStepAccumulator()
        noise_acc = TraderStepAccumulator()
        arb_acc = TraderStepAccumulator()
        sr_cex_execs_this = 0
        sr_dex_execs_this = 0
        delta_a_cex_this = 0.0
        L_pre_trader_this = np.nan
        L_pre_arb_eff_this = np.nan
        dir_arb_this: Optional[str] = None

        def _apply_cex_impact_now(delta_a: float) -> None:
            """
            Apply permanent CEX impact immediately when CEX is touched.

            Updates delta_a_cex_this for series recording, applies impact to ref.m,
            and broadcasts the price move so rebalancer benchmark sees the jump.
            """
            nonlocal delta_a_cex_this
            if abs(delta_a) < 1e-18:
                return
            delta_a_cex_this += delta_a
            ref.apply_impact_only(delta_a)
            _broadcast_price_move(ref.m)

        # --- Mempool structures ---
        mempool_orders = []
        jit_targets: Dict[int, Dict[str, Any]] = {}
        jit_open_positions: Dict[int, Position] = {}
        jit_swap_executed: Dict[int, bool] = {}
        
        # ----- Non-mutating Uni v3 quotes (spacing-aware, can bridge deserts) -----
        def maybe_enqueue_smart_router_intent(m_now: float):
            """
            Enqueue a smart-router swap intent when DEX is competitive; otherwise
            route the trade to the CEX and record its impact and PnL.
            """
            nonlocal trader_y_this, sr_acc, _trader_execs, delta_a_cex_this
            nonlocal total_smart_swaps_executed, smart_swaps_x_to_y, smart_swaps_y_to_x
            nonlocal sr_cex_execs_this
            side = random.choice(["X_to_Y", "Y_to_X"])
            if side == "X_to_Y":
                notional_y = _draw_trader_notional()
                if notional_y <= 0.0:
                    return
                price_snapshot = max(m_now, 1e-18)
                dx = notional_y / price_snapshot
                if dx <= 0.0:
                    return
                initial_quote = pool.quote_x_to_y(dx)
                if initial_quote <= 0.0:
                    return
                # best-ex vs CEX: compare dy_out to dx * m_now (value in token1)
                cex_value_y = dx * m_now
                if initial_quote < theta_T * cex_value_y:
                    # DEX too uncompetitive: execute against CEX
                    # CEX trade is a fair exchange at m_now, so realized PnL = 0
                    dy_cex = cex_value_y
                    trader_steps.append(t); trader_dirs.append("down")
                    smart_activity_steps.append(t); smart_activity_signs.append(+1)
                    delta_y = -dy_cex
                    sr_acc.notional_y += delta_y
                    trader_y_this += delta_y
                    # Record CEX trade PnL directly (fair exchange => PnL = 0)
                    # Don't use record_swap as that would be revalued at settlement
                    sr_acc.record_cex_trade_pnl(0.0)
                    executed = int(dx > 0.0)
                    sr_acc.execs += executed
                    _trader_execs += executed
                    total_smart_swaps_executed += executed
                    smart_swaps_x_to_y += executed
                    sr_cex_execs_this += executed
                    # Sell token0 on CEX => negative Δa_cex (immediate impact)
                    _apply_cex_impact_now(-dx)
                    buffer_log(
                        f"[t={t:03d}] smart CEX swap X_to_Y EXEC dx={dx:.6f} dy_out={dy_cex:.6f} "
                        f"@ m={m_now:.4f} (impact applied)\n"
                    )
                    return
                baseline_quote = _baseline_quote_x_to_y(dx)
                min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
                mempool_orders.append({
                    'type': 'swap',
                    'agent': 'smart',
                    'side': 'X_to_Y',
                    'amount': dx,
                    'unit': 'dx',
                    'm_submit': m_now,
                    'min_output': min_output,
                })
            else:
                dy = _draw_trader_notional()
                if dy <= 0.0:
                    return
                initial_quote = pool.quote_y_to_x(dy)
                if initial_quote <= 0.0:
                    return
                # best-ex vs CEX: compare dx_out to dy / m_now (value in token0)
                dx_cex = dy / max(m_now, 1e-18)
                if initial_quote < theta_T * dx_cex:
                    # DEX too uncompetitive: execute against CEX
                    # CEX trade is a fair exchange at m_now, so PnL is typically 0.
                    trader_steps.append(t); trader_dirs.append("up")
                    smart_activity_steps.append(t); smart_activity_signs.append(-1)
                    sr_acc.notional_y += dy
                    trader_y_this += dy
                    # Record CEX trade PnL directly (fair exchange => PnL = 0)
                    # Don't use record_swap as that would be revalued at settlement
                    sr_acc.record_cex_trade_pnl(0.0)
                    executed = int(dy > 0.0)
                    sr_acc.execs += executed
                    _trader_execs += executed
                    total_smart_swaps_executed += executed
                    smart_swaps_y_to_x += executed
                    sr_cex_execs_this += executed
                    # Buy token0 on CEX => positive Δa_cex (immediate impact)
                    _apply_cex_impact_now(dx_cex)
                    buffer_log(
                        f"[t={t:03d}] smart CEX swap Y_to_X EXEC dy={dy:.6f} dx_out={dx_cex:.6f} "
                        f"@ m={m_now:.4f} (impact applied)\n"
                    )
                    return
                baseline_quote = _baseline_quote_y_to_x(dy)
                min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
                mempool_orders.append({
                    'type': 'swap',
                    'agent': 'smart',
                    'side': 'Y_to_X',
                    'amount': dy,
                    'unit': 'dy',
                    'm_submit': m_now,
                    'min_output': min_output,
                })

        def maybe_enqueue_noise_trader_intent(m_now: float):
            """Enqueue a noise swap intent (no best-ex check)."""
            side = random.choice(["X_to_Y", "Y_to_X"])
            if side == "X_to_Y":
                notional_y = _draw_trader_notional()
                if notional_y > 0.0:
                    price_snapshot = max(m_now, 1e-18)
                    dx = notional_y / price_snapshot
                    if dx > 0.0:
                        initial_quote = pool.quote_x_to_y(dx)
                        if initial_quote <= 0.0:
                            return
                        baseline_quote = _baseline_quote_x_to_y(dx)
                        min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
                        mempool_orders.append({
                            'type': 'swap',
                            'agent': 'noise',
                            'side': 'X_to_Y',
                            'amount': dx,
                            'unit': 'dx',
                            'm_submit': m_now,
                            'min_output': min_output,
                        })
            else:
                dy = _draw_trader_notional()
                if dy > 0.0:
                    initial_quote = pool.quote_y_to_x(dy)
                    if initial_quote <= 0.0:
                        return
                    baseline_quote = _baseline_quote_y_to_x(dy)
                    min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
                    mempool_orders.append({
                        'type': 'swap',
                        'agent': 'noise',
                        'side': 'Y_to_X',
                        'amount': dy,
                        'unit': 'dy',
                        'm_submit': m_now,
                    'min_output': min_output,
                })

        def plan_jiter_targets() -> None:
            """Select the single largest swap intent and mark it for JIT mint/burn."""
            nonlocal jit_targets, jit_swap_executed
            if not jiter_enabled or jiter_agent is None:
                return
            if N_jit <= 0 or liquidity_perc_jit <= 0.0:
                return
            if random.random() >= p_jit:
                return
            swap_orders = [o for o in mempool_orders if o.get("type") == "swap"]
            if not swap_orders:
                return
            # Sort by input amount normalized to token1 using current CEX reference price.
            price_ref = max(cex_ref_for_agents, 1e-18)
            sorted_swaps = sorted(
                swap_orders,
                key=lambda o: (
                    float(o.get("amount", 0.0)) * price_ref
                    if o.get("unit") == "dx"
                    else float(o.get("amount", 0.0))
                ),
                reverse=True,
            )
            # New simplified JIT: always target the single largest swap (N_jit is kept
            # as an enable/disable knob but does not increase target count).
            targets = sorted_swaps[:1]
            for o in targets:
                target_id = id(o)
                o["jit_target"] = target_id
                jit_targets[target_id] = {
                    "side": o.get("side"),
                    "amount": float(o.get("amount", 0.0)),
                    "unit": o.get("unit"),
                }
                jit_swap_executed[target_id] = False

        def execute_mempool_orders():
            nonlocal trader_y_this, sr_acc, noise_acc
            nonlocal dex_notional_y_this
            nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
            nonlocal total_smart_swaps_executed, total_smart_swaps_skipped
            P_pre_exec = pool.price
            executed_smart_swaps = 0
            executed_noise_swaps = 0
            executed_lp_events = 0

            def _exec_one(o):
                nonlocal P_pre_exec, trader_y_this, sr_acc, noise_acc
                nonlocal dex_notional_y_this
                nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
                nonlocal total_smart_swaps_executed, total_smart_swaps_skipped
                nonlocal smart_swaps_x_to_y, smart_swaps_y_to_x
                nonlocal noise_swaps_x_to_y, noise_swaps_y_to_x
                nonlocal executed_smart_swaps, executed_noise_swaps, executed_lp_events
                nonlocal arb_y_this, L_pre_arb_eff_this, dir_arb_this, delta_a_cex_this, _arb_execs
                nonlocal total_arb_swaps_executed, total_arb_no_op_in_band, total_arb_swaps_rejected_profitability
                nonlocal total_jit_trades_executed
                nonlocal jit_open_positions, jit_swap_executed
                nonlocal sr_dex_execs_this
                P_pre_exec = pool.price
                tick_before_exec = pool.tick
                # Execution-time CEX price used for **settlement / cash conversion** in LP
                # mint/burn accounting. Agents still choose ranges off the validated
                # snapshot (`agent_S_ref`, `cex_ref_for_agents`).
                m_exec = float(ref.m)
                typ = o.get('type')

                if typ == "jit_mint":
                    if not jiter_enabled or jiter_agent is None:
                        return
                    tgt = o.get("target_id")
                    plan = jit_targets.get(tgt)
                    if plan is None:
                        return
                    side = plan.get("side")
                    amount_in = float(plan.get("amount", 0.0))
                    if side not in ("X_to_Y", "Y_to_X") or amount_in <= 0.0:
                        return

                    tick_now = pool.tick
                    spacing = int(pool.tick_spacing)
                    S_now = pool.S

                    # When the DEX price is exactly on a band boundary, the next swap in the
                    # corresponding direction will immediately cross into the adjacent band.
                    # If we mint in the "wrong" band here, the required-liquidity estimate
                    # can blow up (denominator ~ 0) and, worse, repeated add/remove of an
                    # astronomically large L can destroy small baseline `liquidity_net`
                    # deltas via catastrophic cancellation.
                    band_lower = tick_now
                    sa_band = pool.s_lower(band_lower)
                    sb_band = pool.s_upper(band_lower)
                    band_scale = max(1.0, abs(sa_band), abs(sb_band), abs(S_now))
                    boundary_tol = EPS_BOUNDARY * band_scale
                    if side == "Y_to_X" and S_now >= sb_band - boundary_tol:
                        band_lower += spacing
                        sa_band = pool.s_lower(band_lower)
                        sb_band = pool.s_upper(band_lower)
                    elif side == "X_to_Y" and S_now <= sa_band + boundary_tol:
                        band_lower -= spacing
                        sa_band = pool.s_lower(band_lower)
                        sb_band = pool.s_upper(band_lower)

                    lower = band_lower
                    upper = band_lower + spacing
                    if upper <= lower:
                        return
                    sa, sb = pool.s_lower(lower), pool.s_lower(upper)

                    L_existing_band = pool.bidx.active_liquidity_at_tick(lower)
                    L_existing_band = max(EPS_LIQ, L_existing_band)

                    def _disable_jit_target(target_id: Any) -> None:
                        """Remove bookkeeping so this swap doesn't count as a JIT success."""
                        try:
                            tgt_id_int = int(target_id)
                        except (TypeError, ValueError):
                            return
                        jit_targets.pop(tgt_id_int, None)
                        jit_swap_executed.pop(tgt_id_int, None)

                    # --- Closed-form JIT sizing ----------------------------------------------
                    # Choose L_target (added liquidity) to maximize expected profit:
                    #   Pi(L) = expected_fee_capture_y(L) - flash_fee_y(L)
                    # where fee capture is limited to the portion of the swap that stays inside
                    # this single tick band, and flash fee is proportional to the token1 value
                    # of minted principal.
                    fee_rate = float(getattr(pool, "f", 0.0))
                    r_t = float(getattr(pool, "r", 1.0 - fee_rate))
                    if fee_rate <= 0.0 or r_t <= 0.0:
                        _disable_jit_target(tgt)
                        return

                    # Limit Jiter to at most `liquidity_perc_jit` share of the tick after minting.
                    q_max = float(liquidity_perc_jit)
                    share_factor_max = q_max / max(1e-12, (1.0 - q_max))
                    L_share_cap = share_factor_max * L_existing_band

                    # Numerical safety: avoid minting astronomically large liquidity (float stability).
                    max_jit_mult = 1e6
                    L_cap_numeric = max_jit_mult * max(1.0, L_existing_band)
                    L_max = min(L_share_cap, L_cap_numeric)
                    if L_max <= 0.0:
                        _disable_jit_target(tgt)
                        return

                    # k_cap is the pre-fee input (in swap input units) that can be processed inside
                    # this tick per unit of total liquidity L_total.
                    denom = 0.0
                    denom_floor = 0.0
                    v_fee = 1.0  # convert input-token fees to token1
                    if side == "Y_to_X":
                        denom = sb_band - S_now
                        denom_floor = boundary_tol
                        v_fee = 1.0
                    else:
                        denom = (1.0 / sa_band) - (1.0 / S_now)
                        inv_scale = max(1.0, abs(1.0 / sa_band), abs(1.0 / S_now))
                        denom_floor = EPS_BOUNDARY * inv_scale
                        v_fee = float(m_exec)
                    if denom <= denom_floor:
                        _disable_jit_target(tgt)
                        return
                    k_cap = denom / r_t
                    if k_cap <= 0.0:
                        _disable_jit_target(tgt)
                        return

                    # Flash fee is linear in liquidity for a fixed tick band:
                    # minted_amounts_at_S(L) is linear in L, so principal_value_y(L) = L * principal_value_y(1).
                    amt0_unit, amt1_unit = minted_amounts_at_S(1.0, sa, sb, S_now)
                    principal_value_per_L_y = float(amt0_unit) * float(m_exec) + float(amt1_unit)
                    c_flash = float(jit_flash_loan_fee) * max(0.0, principal_value_per_L_y)

                    # Convenience scalars:
                    # A = total fee value  if the whole swap stays inside the tick.
                    # b = fee value captured per unit L when the swap crosses out of the tick.
                    A_fee_full = amount_in * fee_rate * v_fee
                    b_fee_per_L_crossing = fee_rate * v_fee * k_cap

                    # Liquidity threshold (added L) needed so the entire swap stays within the tick:
                    # amount_in <= k_cap * (L_existing + L_target)
                    L_full = max(0.0, (amount_in / k_cap) - L_existing_band)

                    L_target = 0.0
                    if c_flash <= 0.0:
                        # No financing cost: maximize fee capture by minting up to the cap.
                        L_target = L_max
                    else:
                        slope_cross = b_fee_per_L_crossing - c_flash
                        if slope_cross <= 0.0:
                            # Even in the best case (swap immediately crosses), fee capture per unit L
                            # does not cover financing cost per unit L => never profitable.
                            L_target = 0.0
                        elif L_max <= L_full:
                            # Swap still crosses at the cap; profit is linear in L in this regime.
                            L_target = L_max
                        else:
                            # In the full-in-range regime, Pi(L) = A * L/(E+L) - cL has an interior maximizer:
                            #   L* = sqrt(A*E/c) - E.
                            # Constrain to L >= L_full and L <= L_max.
                            L_star = math.sqrt((A_fee_full * L_existing_band) / c_flash) - L_existing_band
                            if not math.isfinite(L_star):
                                L_star = 0.0
                            L_target = min(L_max, max(L_full, max(0.0, L_star)))

                    if L_target <= 0.0:
                        _disable_jit_target(tgt)
                        return

                    amt0, amt1 = minted_amounts_at_S(L_target, sa, sb, S_now)

                    # Calculate flash-loan cost BEFORE profitability check (valued at mint-time CEX price).
                    flash_fee_y = 0.0
                    if jit_flash_loan_fee > 0.0:
                        borrowed_value_y = float(amt0) * float(m_exec) + float(amt1)
                        if borrowed_value_y > 0.0:
                            flash_fee_y = borrowed_value_y * float(jit_flash_loan_fee)

                    # --- Profitability filter (expected fees inside tick) ----------------------
                    L_total_tick = L_existing_band + L_target
                    fee_share = float(L_target / max(1e-12, L_total_tick))
                    in_range_input = min(amount_in, k_cap * L_total_tick)
                    fee_total_value_y = in_range_input * fee_rate * v_fee
                    expected_fee_capture_y = fee_share * fee_total_value_y
                    expected_profit_y = expected_fee_capture_y - flash_fee_y
                    if expected_profit_y <= 0.0:
                        _disable_jit_target(tgt)
                        buffer_log(
                            f"[t={t:03d}] JIT SKIP (expected_profit<=0) "
                            f"expected_fee={expected_fee_capture_y:.6g} flash_fee={flash_fee_y:.6g} "
                            f"share={fee_share:.3f} in_range_in={in_range_input:.6g} "
                            f"notional_in={amount_in:.6g} side={side}\n"
                        )
                        return
                    # --------------------------------------------------------------------------

                    # Only apply flash fee AFTER confirming the mint will proceed
                    jiter_agent.flash_fees_paid_y = float(getattr(jiter_agent, "flash_fees_paid_y", 0.0)) + flash_fee_y  # type: ignore[attr-defined]
                    pos = Position(
                        owner=jiter_agent.id,
                        lower=lower,
                        upper=upper,
                        L=L_target,
                        sa=sa,
                        sb=sb,
                        amt0_init=float(amt0),
                        amt1_init=float(amt1),
                        hodl0_value_y=float(amt0) * cex_ref_for_agents + float(amt1),
                    )
                    # Treat JIT as flash-funded: allow token wallets to go negative.
                    jiter_agent.wallet_x = float(getattr(jiter_agent, "wallet_x", 0.0)) - float(amt0)
                    jiter_agent.wallet_y = float(getattr(jiter_agent, "wallet_y", 0.0)) - float(amt1) - flash_fee_y
                    pool.add_liquidity_range(lower, upper, L_target)
                    jiter_agent.positions.append(pos)
                    _register_position(pos)
                    _assert_active_liquidity_state_fast("jit_mint")
                    jiter_agent.L_live = getattr(jiter_agent, "L_live", 0.0) + L_target
                    jit_open_positions[int(tgt)] = pos
                    _rebalance_lp_to_target(jiter_agent, ref.m, pool.S)
                    # Flash fee is an external financing cost: it is already deducted from
                    # Jiter wealth via wallet_y above, and should NOT be booked into the
                    # rebalancing benchmark (otherwise it would cancel in hedged PnL).
                    mint_steps.append(t)
                    mint_sizes.append(L_target)
                    mint_widths.append(upper - lower)
                    mint_is_passive.append(False)
                    mint_is_jiter.append(True)
                    buffer_log(
                        f"[t={t:03d}] JIT MINT L={L_target:.4f} [{lower},{upper}) | "
                        f"L_active={pool.L_active:.4f} | tick={pool.tick} | flash_fee_y={flash_fee_y:.6g}\n"
                    )
                    executed_lp_events += 1
                    return

                if typ == "jit_burn":
                    if not jiter_enabled or jiter_agent is None:
                        return
                    tgt = o.get("target_id")
                    pos = jit_open_positions.pop(int(tgt), None)
                    if pos is None:
                        return
                    try:
                        jiter_agent.positions.remove(pos)
                    except ValueError:
                        pass
                    amt0, amt1 = pos.current_amounts(pool.S)
                    amt0_total = float(amt0) + float(pos.fees0)
                    amt1_total = float(amt1) + float(pos.fees1)
                    # JIT burn + netting: withdraw principal+fees, then convert the net token0
                    # inventory to token1 at the current CEX price, applying immediate CEX impact.
                    # This mirrors regular LP burns, but must account for the flash-funded
                    # (potentially negative) wallet_x carried from the mint.
                    m_exec_jit = float(ref.m)  # Use current CEX price for conversion
                    wallet_x_before = float(getattr(jiter_agent, "wallet_x", 0.0))
                    wallet_y_before = float(getattr(jiter_agent, "wallet_y", 0.0))
                    wallet_x_after = wallet_x_before + amt0_total
                    wallet_y_after = wallet_y_before + amt1_total
                    amt0_net_cex = wallet_x_after
                    # Convert net token0 to token1 (sell if positive, buy if negative)
                    if abs(amt0_net_cex) > 1e-18:
                        wallet_y_after += amt0_net_cex * m_exec_jit
                        delta_a_cex_this += -amt0_net_cex
                        ref.apply_impact_only(-amt0_net_cex)
                        _broadcast_price_move(ref.m)
                    jiter_agent.wallet_x = 0.0
                    jiter_agent.wallet_y = wallet_y_after
                    _unregister_position(pos)
                    pool.add_liquidity_range(pos.lower, pos.upper, -pos.L)
                    _assert_active_liquidity_state_fast("jit_burn")
                    jiter_agent.L_live = max(0.0, getattr(jiter_agent, "L_live", 0.0) - pos.L)
                    _rebalance_lp_to_target(jiter_agent, ref.m, pool.S)
                    burn_steps.append(t)
                    burn_sizes.append(pos.L)
                    burn_is_passive.append(False)
                    burn_is_jiter.append(True)
                    if jit_swap_executed.pop(int(tgt), False):
                        jiter_activity_steps.append(t)
                        jiter_activity_signs.append(+1)
                        total_jit_trades_executed += 1
                    buffer_log(
                        f"[t={t:03d}] JIT BURN L={pos.L:.4f} [{pos.lower},{pos.upper}) | L_active={pool.L_active:.4f} | tick={pool.tick} | "
                        f"amt0_net_cex={amt0_net_cex:+.6f} @ m={m_exec_jit:.4f}\n"
                    )
                    executed_lp_events += 1
                    return

                # Handle LP intents (they don't have 'agent' or 'side')
                if typ in ('lp_burn','lp_mint','lp_recenter'):
                    lp = lp_lookup.get(o.get('lp_id'))
                    if lp is None:
                        return
                    if typ == 'lp_burn':
                        idx = None
                        for i, pos in enumerate(lp.positions):
                            if pos.lower == o.get('lower') and pos.upper == o.get('upper') and abs(pos.L - float(o.get('L', 0.0))) < 1e-12:
                                idx = i; break
                        if idx is None:
                            return
                        burn_any(lp, idx, m_ref=m_exec)
                        executed_lp_events += 1
                        return
                    if typ == 'lp_mint':
                        lower = int(o.get("lower"))
                        upper = int(o.get("upper"))
                        eta = float(o.get("eta", 1.0))
                        created = _execute_cash_budgeted_mint(
                            lp=lp,
                            lower=lower,
                            upper=upper,
                            eta=eta,
                            S_ref=agent_S_ref,
                            m_ref=m_exec,
                            log_prefix=f"LP{lp.id} MINT",
                            is_jiter=False,
                        )
                        if created is None:
                            return
                        executed_lp_events += 1
                        return
                    if typ == 'lp_recenter':
                        idx = None
                        for i, pos in enumerate(lp.positions):
                            if pos.lower == o.get('old_lower') and pos.upper == o.get('old_upper') and abs(pos.L - float(o.get('old_L', 0.0))) < 1e-12:
                                idx = i; break
                        if idx is None:
                            return
                        burn_any(lp, idx, m_ref=m_exec)
                        lower = int(o.get("new_lower"))
                        upper = int(o.get("new_upper"))
                        eta = float(o.get("eta", 1.0))
                        created = _execute_cash_budgeted_mint(
                            lp=lp,
                            lower=lower,
                            upper=upper,
                            eta=eta,
                            S_ref=agent_S_ref,
                            m_ref=m_exec,
                            log_prefix=f"LP{lp.id} RECENTER",
                            is_jiter=False,
                        )
                        if created is None:
                            return
                        executed_lp_events += 1
                        return

                # Handle arbitrage intent (executes before other swaps in a block)
                if typ == 'arb':
                    arb_ref = float(o.get('arb_ref_m', ref.m))
                    # Preview arbitrage profitability (includes liquidity fee + flash loan fee)
                    prev_in, prev_x_out, prev_y_out, prev_dir, _ = preview_arbitrage_to_target(arb_ref)
                    if prev_dir is None:
                        arb_skip_steps.append(t)
                        total_arb_no_op_in_band += 1
                        buffer_log(
                            f"[t={t:03d}] arb NO-OP (in-band): "
                            f"| price={pool.price:.4f} cex={arb_ref:.4f}\n"
                        )
                        return
                    expected_profit = 0.0
                    expected_flash_fee = 0.0
                    if prev_dir == "up":
                        expected_flash_fee = (flash_loan_mult - 1.0) * prev_in
                        expected_profit = prev_x_out * arb_ref - flash_loan_mult * prev_in
                    elif prev_dir == "down":
                        notional_y = prev_in * arb_ref
                        expected_flash_fee = (flash_loan_mult - 1.0) * notional_y
                        expected_profit = prev_y_out - flash_loan_mult * notional_y
                    if expected_profit <= 0.0:
                        arb_skip_steps.append(t)
                        total_arb_swaps_rejected_profitability += 1
                        buffer_log(
                            f"[t={t:03d}] arb SKIPPED (unprofitable): dir={prev_dir} "
                            f"expected_profit={expected_profit:.6f} flash_fee={expected_flash_fee:.6f} "
                            f"| price={pool.price:.4f} cex={arb_ref:.4f}\n"
                        )
                        return

                    price_before = pool.price
                    tick_before = pool.tick
                    in_used, x_out_from_dex, y_out_from_dex, dir_arb, L_first = arbitrage_to_target(arb_ref)
                    if in_used > 0 and dir_arb is not None:
                        L_pre_arb_eff_this = L_first
                        dir_arb_this = dir_arb
                        arb_steps.append(t); arb_dirs.append(dir_arb)

                        if dir_arb == "up":
                            # DEX cheap: buy token0 on DEX, sell on CEX (immediate impact)
                            arb_y_this = +in_used
                            dex_notional_y_this += abs(in_used)
                            arb_acc.record_swap(dy_in=flash_loan_mult * in_used, dx_out=x_out_from_dex)
                            _apply_cex_impact_now(-x_out_from_dex)
                            buffer_log(
                                f"[t={t:03d}] arb swap up dy_in={in_used:.6f} dx_out={x_out_from_dex:.6f} "
                                f"| price {price_before:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick} (impact applied)\n"
                            )
                        else:
                            # DEX expensive: sell token0 on DEX, buy on CEX (immediate impact)
                            arb_y_this = -pool.price * in_used
                            dex_notional_y_this += abs(price_before * in_used)
                            arb_acc.record_swap(dx_in=flash_loan_mult * in_used, dy_out=y_out_from_dex)
                            _apply_cex_impact_now(flash_loan_mult * in_used)
                            buffer_log(
                                f"[t={t:03d}] arb swap down dx_in={in_used:.6f} dy_out={y_out_from_dex:.6f} "
                                f"| price {price_before:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick} (impact applied)\n"
                            )
                        executed = int(in_used > 0)
                        _arb_execs += executed
                        total_arb_swaps_executed += executed
                    return

                if o.get('side') == 'X_to_Y':
                    min_output = o.get('min_output')
                    if min_output is not None:
                        final_quote = pool.quote_x_to_y(o['amount'])
                        if final_quote <= min_output:
                            agent = o.get('agent')
                            if agent == 'smart':
                                total_smart_swaps_skipped += 1
                            elif agent == 'noise':
                                total_noise_swaps_skipped += 1
                            buffer_log(
                                f"[t={t:03d}] {agent or 'N/A'} swap X_to_Y SKIPPED (slippage). "
                                f"final_quote={final_quote:.4f} <= min_output={min_output:.4f} | tick={tick_before_exec}\n"
                            )
                            return
                    if pool.L_active <= EPS_LIQ:
                        return
                    S_before = pool.S; tick_before = pool.tick
                    used_dx_pre, dy_out_real, fee_x = pool.swap_x_to_y(o['amount'], fee_cb=allocate_fees)
                    _flush_pending_rebalance()  # Batch rebalance all LPs touched during this swap
                    _assert_active_liquidity_state_fast("mempool_swap_x_to_y")
                    if used_dx_pre <= EPS_LIQ:
                        _rollback_pool_state(tick_before, S_before)
                        return
                    dex_notional_y_this += abs(P_pre_exec * used_dx_pre)
                    executed = int(used_dx_pre > 0)
                    tgt = o.get("jit_target")
                    if executed and tgt is not None:
                        try:
                            tgt_id = int(tgt)
                        except (TypeError, ValueError):
                            tgt_id = None
                        if tgt_id is not None and tgt_id in jit_swap_executed:
                            jit_swap_executed[tgt_id] = True
                    agent = o.get('agent')
                    if agent == 'smart':
                        trader_steps.append(t); trader_dirs.append('down')
                        smart_activity_steps.append(t); smart_activity_signs.append(+1)
                        sr_acc.notional_y += -P_pre_exec * used_dx_pre
                        trader_y_this += -P_pre_exec * used_dx_pre
                        sr_acc.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)
                        sr_acc.execs += int(used_dx_pre > 0)
                        total_smart_swaps_executed += executed
                        smart_swaps_x_to_y += int(used_dx_pre > 0)
                        executed_smart_swaps += executed
                        sr_dex_execs_this += executed
                        buffer_log(
                            f"[t={t:03d}] smart swap X_to_Y EXEC dx={used_dx_pre:.6f} dy_out={dy_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                    elif agent == 'noise':
                        trader_steps.append(t); trader_dirs.append('down')
                        noise_activity_steps.append(t); noise_activity_signs.append(+1)
                        noise_acc.notional_y += -P_pre_exec * used_dx_pre
                        trader_y_this += -P_pre_exec * used_dx_pre
                        noise_acc.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)
                        noise_acc.execs += int(used_dx_pre > 0)
                        total_noise_swaps_executed += executed
                        noise_swaps_x_to_y += int(used_dx_pre > 0)
                        executed_noise_swaps += executed
                        buffer_log(
                            f"[t={t:03d}] noise swap X_to_Y EXEC dx={used_dx_pre:.6f} dy_out={dy_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                else:
                    min_output = o.get('min_output')
                    if min_output is not None:
                        final_quote = pool.quote_y_to_x(o['amount'])
                        if final_quote <= min_output:
                            agent = o.get('agent')
                            if agent == 'smart':
                                total_smart_swaps_skipped += 1
                            elif agent == 'noise':
                                total_noise_swaps_skipped += 1
                            buffer_log(
                                f"[t={t:03d}] {agent or 'N/A'} swap Y_to_X SKIPPED (slippage). "
                                f"final_quote={final_quote:.4f} <= min_output={min_output:.4f} | tick={tick_before_exec}\n"
                            )
                            return
                    if pool.L_active <= EPS_LIQ:
                        return
                    S_before = pool.S; tick_before = pool.tick
                    used_dy_pre, dx_out_real, fee_y = pool.swap_y_to_x(o['amount'], fee_cb=allocate_fees)
                    _flush_pending_rebalance()  # Batch rebalance all LPs touched during this swap
                    _assert_active_liquidity_state_fast("mempool_swap_y_to_x")
                    if used_dy_pre <= EPS_LIQ:
                        _rollback_pool_state(tick_before, S_before)
                        return
                    dex_notional_y_this += abs(used_dy_pre)
                    executed = int(used_dy_pre > 0)
                    tgt = o.get("jit_target")
                    if executed and tgt is not None:
                        try:
                            tgt_id = int(tgt)
                        except (TypeError, ValueError):
                            tgt_id = None
                        if tgt_id is not None and tgt_id in jit_swap_executed:
                            jit_swap_executed[tgt_id] = True
                    agent = o.get('agent')
                    if agent == 'smart':
                        trader_steps.append(t); trader_dirs.append('up')
                        smart_activity_steps.append(t); smart_activity_signs.append(-1)
                        sr_acc.notional_y += +used_dy_pre
                        trader_y_this += +used_dy_pre
                        sr_acc.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)
                        sr_acc.execs += int(used_dy_pre > 0)
                        total_smart_swaps_executed += executed
                        smart_swaps_y_to_x += int(used_dy_pre > 0)
                        executed_smart_swaps += executed
                        sr_dex_execs_this += executed
                        buffer_log(
                            f"[t={t:03d}] smart swap Y_to_X EXEC dy={used_dy_pre:.6f} dx_out={dx_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                    elif agent == 'noise':
                        trader_steps.append(t); trader_dirs.append('up')
                        noise_activity_steps.append(t); noise_activity_signs.append(-1)
                        noise_acc.notional_y += +used_dy_pre
                        trader_y_this += +used_dy_pre
                        noise_acc.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)
                        noise_acc.execs += int(used_dy_pre > 0)
                        total_noise_swaps_executed += executed
                        noise_swaps_y_to_x += int(used_dy_pre > 0)
                        executed_noise_swaps += executed
                        buffer_log(
                            f"[t={t:03d}] noise swap Y_to_X EXEC dy={used_dy_pre:.6f} dx_out={dx_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
            order_book = list(mempool_orders)
            # Ensure arbitrage intents execute first, then shuffle the rest
            arb_orders = [o for o in order_book if o.get('type') == 'arb']
            non_arb_orders = [o for o in order_book if o.get('type') != 'arb']
            random.shuffle(non_arb_orders)
            order_book = list(arb_orders)
            for o in non_arb_orders:
                tgt = o.get("jit_target")
                if tgt is not None and tgt in jit_targets and jiter_agent is not None:
                    order_book.append({"type": "jit_mint", "target_id": tgt})
                    order_book.append(o)
                    order_book.append({"type": "jit_burn", "target_id": tgt})
                else:
                    order_book.append(o)
            tick_before_orders = pool.tick
            buffer_log(
                f"[t={t:03d}] MEMPOOL before P={P_pre_exec:.4f} | tick={tick_before_orders} | "
                f"n_orders={len(order_book)}\n"
            )
            for o in order_book:
                _exec_one(o)
            buffer_log(
                f"[t={t:03d}] MEMPOOL after P={pool.price:.4f} | tick={pool.tick} | "
                f"smart_exec={executed_smart_swaps} | noise_exec={executed_noise_swaps} | "
                f"lp_events={executed_lp_events}\n"
            )
            mempool_orders.clear()

        # --- LP scheduling ---
        # Figure out which LPs are due to act this step.
        due = []
        for i, lp in enumerate(LPs):
            if getattr(lp, "is_jiter", False):
                continue
            if lp.cooldown > 0:
                lp.cooldown -= 1
                # let the review clock keep ticking while cooling down
                lp.next_review = max(1, lp.next_review - 1)
                continue
            lp.next_review -= 1
            if lp.next_review <= 0:
                due.append(i)
                lp.next_review = int(np.random.geometric(lp.review_rate))

        random.shuffle(due)

        def _enable(indices):
            s = set(indices)
            for j, lp in enumerate(LPs):
                lp.can_act = (j in s)

        # ===================== MEMPOOL SCHEDULING & ORDER =====================
        # Freeze the validated snapshot, run micro-steps that diffuse the CEX
        # and enqueue trader intents, then replay the mempool (arb first) with
        # LP intents included.
        # =====================================================================
        arb_ref_m_start = ref.m  # block-start CEX price (equals this block's validated snapshot for agents)
        buffer_log(f"[t={t:03d}] BLOCK start m={arb_ref_m_start:.4f} due_lp={len(due)}\n")
        # Micro-time logging: one point per micro-step, recorded at the end of the
        # diffusion step. The final micro-step of the block includes the full
        # mempool replay, so the logged DEX price reflects the cumulative within-block
        # execution effects and the logged CEX price includes any impact applied
        # during that execution phase.
        for k in range(block_time):
            if smart_lambda_micro_step > 0.0:
                n_smart = int(np.random.poisson(smart_lambda_micro_step))
                for _ in range(n_smart):
                    maybe_enqueue_smart_router_intent(ref.m)  # Use current CEX price for fair exchange
            if noise_lambda_micro_step > 0.0:
                n_noise = int(np.random.poisson(noise_lambda_micro_step))
                for _ in range(n_noise):
                    maybe_enqueue_noise_trader_intent(ref.m)  # Use current CEX price for fair exchange
            ref.diffuse_only()
            _broadcast_price_move(ref.m)
            if k < block_time - 1:
                micro_steps.append(micro_counter)
                P_micro.append(pool.price)
                M_micro.append(ref.m)
                micro_counter += 1

        # --- Arbitrage intent (executes first in mempool) ---
        arb_ref_m = cex_ref_for_agents  # snapshot from end of previous block
        target_band_m = cex_ref_for_agents
        band_lo_target.append(target_band_m * r / flash_loan_mult)
        band_hi_target.append(target_band_m * flash_loan_mult / max(r, 1e-18))
        mempool_orders.append({'type': 'arb', 'arb_ref_m': arb_ref_m})

        # --- Include LP intents in the mempool (shuffled with traders) ---
        # Allow due LPs to act this block
        _enable(due)

        # Burns (TP/SL + passive Poisson targets)
        if passive_burns_per_block > 0.0:
            burnable_positions: List[Tuple[int, int, int, float]] = []
            for lp_idx in due:
                lp = LPs[lp_idx]
                if getattr(lp, "is_jiter", False):
                    continue
                if not lp.can_act or not bool(getattr(lp, "is_passive", False)):
                    continue
                for pos in getattr(lp, "positions", []):
                    burnable_positions.append((lp.id, pos.lower, pos.upper, float(pos.L)))
            if burnable_positions:
                n_burn_intents = int(np.random.poisson(passive_burns_per_block))
                n_burn_intents = min(n_burn_intents, len(burnable_positions))
                if n_burn_intents > 0:
                    for lp_id, lower, upper, L_val in random.sample(burnable_positions, k=n_burn_intents):
                        mempool_orders.append({'type':'lp_burn','lp_id': lp_id,'lower': lower,'upper': upper,'L': L_val})

        for lp_idx in due:
            lp = LPs[lp_idx]
            if getattr(lp, "is_jiter", False):
                continue
            if not lp.can_act:
                continue
            if lp.is_passive:
                continue
            to_burn = []
            for i, pos in enumerate(lp.positions):
                pnl = pos.PnL_y(agent_S_ref, ref.m)
                if pnl >= theta_TP * pos.hodl0_value_y or pnl <= -theta_SL * pos.hodl0_value_y:
                    to_burn.append(i)
            for i in reversed(to_burn):
                pos = lp.positions[i]
                mempool_orders.append({'type':'lp_burn','lp_id': lp.id,'lower': pos.lower,'upper': pos.upper,'L': pos.L})

        # Recenter (narrow LPs that have been out-of-range past their threshold)
        for lp_idx in due:
            lp = LPs[lp_idx]
            if getattr(lp, "is_jiter", False):
                continue
            if not lp.can_act:
                continue
            to_recenters = []
            for i, pos in enumerate(lp.positions):
                in_rng = pos.in_range(agent_tick_ref)
                out_steps = getattr(pos, "out_steps", 0)
                out_steps = 0 if in_rng else out_steps + 1
                setattr(pos, "out_steps", out_steps)
                k_out_thresh = getattr(lp, "k_out_threshold", k_out_max)
                if lp.is_active_narrow and out_steps >= k_out_thresh:
                    to_recenters.append(i)
            for i in reversed(to_recenters):
                pos = lp.positions[i]
                width_ticks = w_ticks
                n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
                S_now = agent_S_ref
                sps = pool.tick_spacing
                nb = n_bands
                denom = (1.0 + (pool.g ** (nb * sps)))
                if denom <= 0.0:
                    denom = 1.0
                lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                lower = pool._snap(int(round(lower_real)))
                upper = lower + nb * sps
                eta = _draw_wallet_utilization_factor()
                mempool_orders.append({'type':'lp_recenter','lp_id': lp.id,
                                       'old_lower': pos.lower,'old_upper': pos.upper,'old_L': pos.L,
                                       'new_lower': lower,'new_upper': upper,'eta': eta})

        def _enqueue_lp_mint(lp: LPAgent, is_passive_mint: bool) -> None:
            if getattr(lp, "cooldown", 0) > 0:
                return
            eta = _draw_wallet_utilization_factor()

            S_now = agent_S_ref
            n_bands: Optional[int] = None
            if is_passive_mint:
                if passive_width_pct is None:
                    width_ticks = max(int(passive_width_ticks), pool.tick_spacing)
                    n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
            else:
                n_bands = max(1, int(round(w_ticks / pool.tick_spacing)))

            if is_passive_mint and passive_width_pct is not None:
                lower, upper = _passive_range_ticks_from_pct(S_now)
            else:
                assert n_bands is not None
                sps = pool.tick_spacing
                nb = n_bands
                denom = (1.0 + (pool.g ** (nb * sps)))
                if denom <= 0.0:
                    denom = 1.0
                lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                lower = pool._snap(int(round(lower_real)))
                upper = lower + nb * sps
                if upper <= lower:
                    upper = lower + pool.tick_spacing

            mempool_orders.append({'type':'lp_mint','lp_id': lp.id,'lower': lower,'upper': upper,'eta': eta})

        # New mints (Poisson targets; respect due/cooldown)
        if narrow_mints_per_block > 0.0:
            narrow_candidates = []
            for lp_idx in due:
                lp = LPs[lp_idx]
                if getattr(lp, "is_jiter", False):
                    continue
                if not lp.can_act or bool(getattr(lp, "is_passive", False)):
                    continue
                if getattr(lp, "cooldown", 0) > 0:
                    continue
                narrow_candidates.append(lp)
            if narrow_candidates:
                n_mints = int(np.random.poisson(narrow_mints_per_block))
                for _ in range(n_mints):
                    _enqueue_lp_mint(random.choice(narrow_candidates), is_passive_mint=False)

        if passive_mints_per_block > 0.0:
            passive_candidates = []
            for lp_idx in due:
                lp = LPs[lp_idx]
                if getattr(lp, "is_jiter", False):
                    continue
                if not lp.can_act or not bool(getattr(lp, "is_passive", False)):
                    continue
                if getattr(lp, "cooldown", 0) > 0:
                    continue
                passive_candidates.append(lp)
            if passive_candidates:
                n_mints = int(np.random.poisson(passive_mints_per_block))
                for _ in range(n_mints):
                    _enqueue_lp_mint(random.choice(passive_candidates), is_passive_mint=True)

        # Execute all mempool intents (traders + LPs) in random order
        L_pre_trader_this = pool.L_active
        plan_jiter_targets()
        execute_mempool_orders()
        micro_steps.append(micro_counter)
        P_micro.append(pool.price)
        M_micro.append(ref.m)
        micro_counter += 1
        if micro_steps:
            micro_valid_steps.append(micro_steps[-1])
            micro_valid_prices.append(pool.price)

        # disable everyone for next step
        _enable([])

        # Align rebalancing benchmark with end-of-block exposures
        # (CEX impact already applied immediately at each CEX touch)
        _rebalance_all(ref.m, pool.S)

        # ---- CEX update (impact already applied; just record settlement price) ----
        # Note: impact is now applied immediately when CEX is touched (arb, smart-router,
        # LP/JIT conversions). No aggregated impact to apply here.

        settlement_m = ref.m
        sr_acc.settle(settlement_m)
        noise_acc.settle(settlement_m)
        trader_pnl_this = sr_acc.pnl + noise_acc.pnl
        arb_acc.settle(arb_ref_m)
        arb_pnl_this = arb_acc.pnl

        _assert_active_liquidity_state_full("end_of_step")

        # ---- Record end-of-step + invariants ----
        P_after = pool.price
        P_series.append(P_after)
        M_series.append(ref.m)
        cex_sigma_series.append(ref.sigma)
        cex_regime_series.append(getattr(ref, "regime_state", "L"))
        delta_a_cex_series.append(delta_a_cex_this)

        x_e, y_e = reserves_in_active_tick()
        X_active_end.append(x_e)
        Y_active_end.append(y_e)
        _val_x = x_e * pool.price
        _val_y = y_e
        _den = max(1e-12, (_val_x + _val_y))
        fee_imb_series.append((_val_y - _val_x) / _den)

        L_end.append(pool.L_active)
        # ---- PnL bookkeeping ----
        trader_pnl_steps.append(trader_pnl_this)
        arb_pnl_steps.append(arb_pnl_this)
        trader_exec_count.append(_trader_execs)
        arb_exec_count.append(_arb_execs)
        lp_total = 0.0                 # hedged PnL = V^{LP} - V^{reb}
        lp_total_active = 0.0          # active narrow LPs
        lp_total_passive = 0.0         # passive LPs
        lp_unhedged_total = 0.0        # V^{LP} - V^{LP}_0
        lp_unhedged_active = 0.0
        lp_unhedged_passive = 0.0
        lp_rebal_total = 0.0           # benchmark PnL (∫ x_t dM_t)
        lp_rebal_active = 0.0
        lp_rebal_passive = 0.0
        lp_rebal_value_total = 0.0     # V^{reb}_t
        lp_rebal_value_active = 0.0
        lp_rebal_value_passive = 0.0
        lp_fee_value_total = 0.0       # F_t (cumulative fees, marked to m_t)
        lp_fee_value_active = 0.0
        lp_fee_value_passive = 0.0
        lp_fees0_earned_total = 0.0    # cumulative token0 fees earned (no MtM)
        lp_fees1_earned_total = 0.0    # cumulative token1 fees earned (no MtM)
        lp_fees0_earned_active = 0.0
        lp_fees1_earned_active = 0.0
        lp_fees0_earned_passive = 0.0
        lp_fees1_earned_passive = 0.0
        lp_wallet_total = 0.0
        lp_wallet_active = 0.0
        lp_wallet_passive = 0.0
        lp_wealth_total = 0.0
        lp_wealth_active = 0.0
        lp_wealth_passive = 0.0
        jiter_wallet_now = 0.0
        jiter_fee_value_now = 0.0
        jiter_fees0_earned_now = 0.0
        jiter_fees1_earned_now = 0.0
        jiter_position_value_now = 0.0
        jiter_wealth_now = 0.0
        jiter_pnl_now = 0.0
        jiter_flash_fee_paid_now = 0.0
        for lp in LPs:
            # Seed/background LPs provide liquidity but are excluded from
            # cohort-level PnL and wealth statistics.
            if getattr(lp, "is_jiter", False):
                _ensure_rebalancer_initialized(lp, ref.m, pool.S)
                jiter_wallet_now = (
                    float(getattr(lp, "wallet_x", 0.0)) * float(ref.m)
                    + float(getattr(lp, "wallet_y", 0.0))
                )
                jiter_fee_value_now = lp_total_fee_earned_value_y(lp, ref.m)
                jiter_fees0_earned_now = float(getattr(lp, "fees0_earned", 0.0))
                jiter_fees1_earned_now = float(getattr(lp, "fees1_earned", 0.0))
                jiter_position_value_now = lp_total_position_value_y(lp, pool.S, ref.m)
                jiter_wealth_now = lp_wealth_y(lp, pool.S, ref.m)
                jiter_flash_fee_paid_now = float(getattr(lp, "flash_fees_paid_y", 0.0))
                rb = lp.rebalancer
                jiter_rebal_value_now = rb.initial_rebal_value_y + rb.cumulative_R
                # Hedged PnL = V^LP - V^reb (matches LP cohort sign convention)
                jiter_pnl_now = jiter_wealth_now - jiter_rebal_value_now
                continue
            if getattr(lp, "is_seed", False):
                continue

            wallet_value_y = (
                float(getattr(lp, "wallet_x", 0.0)) * float(ref.m)
                + float(getattr(lp, "wallet_y", 0.0))
            )
            lp_wallet_total += wallet_value_y
            rb = lp.rebalancer
            _ensure_rebalancer_initialized(lp, ref.m, pool.S)
            wealth_now = lp_wealth_y(lp, pool.S, ref.m)
            fee_value_now = lp_total_fee_earned_value_y(lp, ref.m)
            rebal_value_now = rb.initial_rebal_value_y + rb.cumulative_R
            hedged_pnl = wealth_now - rebal_value_now
            unhedged_pnl = wealth_now - rb.initial_lp_value_y
            rb.hedged_pnl_cum = hedged_pnl
            rb.last_wealth_y = wealth_now
            rb.last_cumulative_R = rb.cumulative_R
            rb.last_M = ref.m

            lp_total += hedged_pnl
            lp_unhedged_total += unhedged_pnl
            lp_rebal_total += rb.cumulative_R
            lp_rebal_value_total += rebal_value_now
            lp_fee_value_total += fee_value_now
            lp_wealth_total += wealth_now
            fees0_earned = float(getattr(lp, "fees0_earned", 0.0))
            fees1_earned = float(getattr(lp, "fees1_earned", 0.0))
            lp_fees0_earned_total += fees0_earned
            lp_fees1_earned_total += fees1_earned

            if lp.is_passive:
                lp_total_passive += hedged_pnl
                lp_unhedged_passive += unhedged_pnl
                lp_rebal_passive += rb.cumulative_R
                lp_rebal_value_passive += rebal_value_now
                lp_fee_value_passive += fee_value_now
                lp_fees0_earned_passive += fees0_earned
                lp_fees1_earned_passive += fees1_earned
                lp_wallet_passive += wallet_value_y
                lp_wealth_passive += wealth_now
            elif lp.is_active_narrow:
                lp_total_active += hedged_pnl
                lp_unhedged_active += unhedged_pnl
                lp_rebal_active += rb.cumulative_R
                lp_rebal_value_active += rebal_value_now
                lp_fee_value_active += fee_value_now
                lp_fees0_earned_active += fees0_earned
                lp_fees1_earned_active += fees1_earned
                lp_wallet_active += wallet_value_y
                lp_wealth_active += wealth_now
        lp_lvr_total = lp_fee_value_total - lp_total
        lp_lvr_active = lp_fee_value_active - lp_total_active
        lp_lvr_passive = lp_fee_value_passive - lp_total_passive
        lp_pnl_total_series.append(lp_total)
        lp_pnl_active_series.append(lp_total_active)
        lp_pnl_passive_series.append(lp_total_passive)
        lp_unhedged_total_series.append(lp_unhedged_total)
        lp_unhedged_active_series.append(lp_unhedged_active)
        lp_unhedged_passive_series.append(lp_unhedged_passive)
        lp_rebal_total_series.append(lp_rebal_total)
        lp_rebal_active_series.append(lp_rebal_active)
        lp_rebal_passive_series.append(lp_rebal_passive)
        lp_rebal_value_total_series.append(lp_rebal_value_total)
        lp_rebal_value_active_series.append(lp_rebal_value_active)
        lp_rebal_value_passive_series.append(lp_rebal_value_passive)
        lp_fee_value_total_series.append(lp_fee_value_total)
        lp_fee_value_active_series.append(lp_fee_value_active)
        lp_fee_value_passive_series.append(lp_fee_value_passive)
        lp_fees0_earned_total_series.append(lp_fees0_earned_total)
        lp_fees1_earned_total_series.append(lp_fees1_earned_total)
        lp_fees0_earned_active_series.append(lp_fees0_earned_active)
        lp_fees1_earned_active_series.append(lp_fees1_earned_active)
        lp_fees0_earned_passive_series.append(lp_fees0_earned_passive)
        lp_fees1_earned_passive_series.append(lp_fees1_earned_passive)
        lp_lvr_total_series.append(lp_lvr_total)
        lp_lvr_active_series.append(lp_lvr_active)
        lp_lvr_passive_series.append(lp_lvr_passive)
        jiter_wallet_series.append(jiter_wallet_now)
        jiter_wealth_series.append(jiter_wealth_now)
        jiter_fee_value_series.append(jiter_fee_value_now)
        jiter_fees0_earned_series.append(jiter_fees0_earned_now)
        jiter_fees1_earned_series.append(jiter_fees1_earned_now)
        jiter_position_value_series.append(jiter_position_value_now)
        jiter_pnl_series.append(jiter_pnl_now)
        jiter_flash_fee_paid_series.append(jiter_flash_fee_paid_now)

        # ================== Dynamic fee controller  ==================
        # By default, signals are based on END-OF-STEP state and the new
        # fee applies NEXT step.

        # 1) Volatility signal (squared log-return)
        # Compute for both CEX and DEX prices; use the appropriate one based on fee_mode
        try:
            # CEX volatility (for volatility_cex mode)
            log_cex_now = math.log(max(ref.m, 1e-18))
            log_cex_prev = math.log(max(prev_cex_for_vol, 1e-18))
            vol_obs_cex = (log_cex_now - log_cex_prev)**2
            # DEX volatility (for volatility_dex mode)
            log_dex_now = math.log(max(pool.price, 1e-18))
            log_dex_prev = math.log(max(prev_dex_for_vol, 1e-18))
            vol_obs_dex = (log_dex_now - log_dex_prev)**2
        except ValueError:
            _vprint("[fee_mode] ValueError in log computation for volatility observation")
            vol_obs_cex = 0.0
            vol_obs_dex = 0.0
        prev_cex_for_vol = ref.m
        prev_dex_for_vol = pool.price
        # Select which volatility observation to use based on fee_mode
        if fee_mode == "volatility_dex":
            vol_obs = vol_obs_dex
        else:
            vol_obs = vol_obs_cex  # default to CEX for volatility_cex and other modes
        sigma_hat_ewma = ewma_sigma_fee.update(vol_obs)

        # 2) Toxicity / basis (fee-adjusted log gap)
        fee_band_ln = -math.log1p(-pool.f)  # ln(1/(1-f))
        log_gap = abs(math.log(max(pool.price, 1e-18)) - math.log(max(ref.m, 1e-18)))
        B_obs = max(0.0, log_gap - fee_band_ln)
        B_hat = ewma_basis_fee.update(B_obs)
        basis_ticks = B_hat / TICK_LN   # convert log-gap to "ticks"

        # 3) LVR-gap (dLVR - dFees), normalized by DEX notional 
        delta_fee_value_total = lp_fee_value_total - prev_lp_fee_value_total
        delta_lvr_total = lp_lvr_total - prev_lp_lvr_total
        prev_lp_fee_value_total = lp_fee_value_total
        prev_lp_lvr_total = lp_lvr_total
        lvr_gap_signal = ewma_lvr_gap_fee.v
        lvr_gap_signal_valid = dex_notional_y_this > 1e-18
        if lvr_gap_signal_valid:
            lvr_gap_obs = (delta_lvr_total - delta_fee_value_total) / dex_notional_y_this
            lvr_gap_signal = ewma_lvr_gap_fee.update(lvr_gap_obs)

        # Record raw signals for diagnostics/plotting.
        # For volatility-based fee mode, fee_sigma_series tracks EWMA(|log-return|).
        sigma_signal = sigma_hat_ewma
        fee_sigma_series.append(sigma_signal)
        fee_basis_ticks_series.append(basis_ticks)

        # Select controller
        f_raw = pool.f
        stage_update = True
        if fee_mode in ("volatility_cex", "volatility_dex"):
            f_raw = k_sigma * np.sqrt(sigma_signal)
        elif fee_mode == "toxicity":
            f_raw = k_basis * basis_ticks
        elif fee_mode == "lvr_fee_ewma":
            # Feedback controller around the current fee.
            if not lvr_gap_signal_valid:
                f_raw = pool.f
                stage_update = False
            else:
                f_raw = pool.f + k_lvr * lvr_gap_signal
        else:
            f_raw = pool.f  # "static": no change

        # Controller signal used for plotting (depends on fee_mode)
        if fee_mode in ("volatility_cex", "volatility_dex"):
            ctrl_sig = sigma_signal
        elif fee_mode == "toxicity":
            ctrl_sig = basis_ticks
        elif fee_mode == "lvr_fee_ewma":
            ctrl_sig = lvr_gap_signal
        else:
            ctrl_sig = 0.0
        fee_signal_series.append(ctrl_sig)

        # Clip and apply hysteresis (min/max step in bps, cooldown).
        if stage_update:
            # f_tgt = clamp(f_raw, f_min, f_max)
            min_step = fee_step_bps_min / 1e4
            max_step = fee_step_bps_max / 1e4
            delta_f = f_raw - pool.f
            if abs(delta_f) >= min_step:
                step = math.copysign(min(abs(delta_f), max_step), delta_f)
                f_new = clamp(pool.f + step, f_min, f_max)
                if abs(f_new - pool.f) >= 1e-12:
                    fee_next = f_new
                    # Only start the cooldown when scheduling a new change; otherwise countdown would restart every step.
                    if fee_cooldown_left <= 0:
                        fee_cooldown_left = max(0, int(fee_cooldown))

        # record current fee (before next-step commit)
        fee_series.append(pool.f)
        # ==================================================================

        # store per-step trader/arb details (now that order is randomized)
        trader_y_series.append(trader_y_this)
        arb_y_series.append(arb_y_this)
        L_pre_trader.append(L_pre_trader_this)

        sr_y_series.append(sr_acc.notional_y)
        noise_y_series.append(noise_acc.notional_y)
        dex_notional_y_series.append(dex_notional_y_this)
        sr_pnl_steps.append(sr_acc.pnl)
        noise_pnl_steps.append(noise_acc.pnl)
        sr_exec_count.append(sr_acc.execs)
        noise_exec_count.append(noise_acc.execs)
        sr_cex_exec_count.append(sr_cex_execs_this)
        sr_dex_exec_count.append(sr_dex_execs_this)
        L_pre_arb_eff.append(L_pre_arb_eff_this)

        price_moved = abs(P_after - P_before) > EPS_PRICE_CHANGE
        had_fill = (abs(trader_y_this) > 0) or (abs(arb_y_this) > 0)
        had_L_event = (t in mint_steps) or (t in burn_steps)
        if price_moved and not (had_fill or had_L_event):
            raise RuntimeError(
                f"DEX price changed at t={t} without swap or LP ΔL. L_active={pool.L_active:.4f}. Change {abs(P_after - P_before)}"
            )

        # Save verbose step info to a txt file in abm_results
        log_line = (
        f"[t={t:03d}] DEX={pool.price:.4f} | CEX={ref.m:.4f} | "
        f"traderY={trader_y_this:.2f} | arb_dir={dir_arb_this} arbY={arb_y_this:.2f} | "
        f"L={pool.L_active:.4f} | tick={pool.tick} | w_ticks={w_ticks}"
        )
        buffer_log(log_line + "\n")

        if liquidity_for_gif:
            liq_history.append(dict(pool.liquidity_net))
        tick_history.append(pool.tick)

        validated_S = pool.S
        validated_tick = pool.tick
        validated_cex = ref.m

    # Smart-router DEX share, computed every n_block_SR_ratio blocks:
    # ratio = DEX / (CEX + DEX), where counts are executed trades routed to each venue.
    sr_dex_share_steps: List[int] = []
    sr_dex_share_series: List[float] = []
    sr_cex_count_by_window: List[int] = []
    sr_dex_count_by_window: List[int] = []

    total_sr_cex_execs = int(sum(sr_cex_exec_count))
    total_sr_dex_execs = int(sum(sr_dex_exec_count))
    total_sr_execs = total_sr_cex_execs + total_sr_dex_execs
    sr_dex_share_overall = (
        float(total_sr_dex_execs / total_sr_execs) if total_sr_execs > 0 else float("nan")
    )

    for w_start in range(0, len(sr_cex_exec_count), n_block_SR_ratio):
        w_end = min(w_start + n_block_SR_ratio, len(sr_cex_exec_count))
        w_cex = int(sum(sr_cex_exec_count[w_start:w_end]))
        w_dex = int(sum(sr_dex_exec_count[w_start:w_end]))
        w_total = w_cex + w_dex
        ratio = float(w_dex / w_total) if w_total > 0 else float("nan")
        sr_dex_share_steps.append(w_end - 1)
        sr_dex_share_series.append(ratio)
        sr_cex_count_by_window.append(w_cex)
        sr_dex_count_by_window.append(w_dex)

    sr_dex_share_arr = np.asarray(sr_dex_share_series, dtype=float)
    if sr_dex_share_arr.size > 0 and np.isfinite(sr_dex_share_arr).any():
        sr_dex_share_mean = float(np.nanmean(sr_dex_share_arr))
    else:
        sr_dex_share_mean = float("nan")

    if not light_mode:
        summary_lines = [
            "# Run summary",
            f"total_mints = {len(mint_steps)}",
            f"total_burns = {len(burn_steps)}",
            f"arb_trades_executed = {total_arb_swaps_executed}",
            f"arb_trades_no_op_in_band = {total_arb_no_op_in_band}",
            f"arb_trades_rejected_profitability = {total_arb_swaps_rejected_profitability}",
            f"jit_trades_executed = {total_jit_trades_executed}",
            f"total_noise_trader_swaps = {total_noise_swaps_executed}",
            f"noise_trader_swaps_rejected_slippage = {total_noise_swaps_skipped}",
            f"total_smart_router_swaps = {total_smart_swaps_executed}",
            f"smart_router_swaps_rejected_slippage = {total_smart_swaps_skipped}",
            f"smart_router_swaps_cex_routed = {total_sr_cex_execs}",
            f"smart_router_swaps_dex_routed = {total_sr_dex_execs}",
            f"smart_router_dex_share_overall = {sr_dex_share_overall:.6f}",
            f"smart_router_dex_share_mean = {sr_dex_share_mean:.6f}",
            f"n_block_SR_ratio = {n_block_SR_ratio}",
            f"smart_router_swaps_X_to_Y (price down) = {smart_swaps_x_to_y}",
            f"smart_router_swaps_Y_to_X (price up) = {smart_swaps_y_to_x}",
            f"noise_trader_swaps_X_to_Y (price down) = {noise_swaps_x_to_y}",
            f"noise_trader_swaps_Y_to_X (price up) = {noise_swaps_y_to_x}",
            "----------------------------------------------------------\n",
        ]

        flush_log_buffer()
        assert verbose_log is not None
        verbose_log.flush()
        verbose_log.close()
        verbose_path = Path(verbose_log_path_str)
        try:
            original_text = verbose_path.read_text()
        except FileNotFoundError:
            original_text = ""
        verbose_path.write_text("\n".join(summary_lines) + original_text)

    if light_mode:
        sr_pnl_steps_arr = np.asarray(sr_pnl_steps, dtype=float)
        noise_pnl_steps_arr = np.asarray(noise_pnl_steps, dtype=float)
        arb_pnl_steps_arr = np.asarray(arb_pnl_steps, dtype=float)
        return {
            "smart_router_pnl_cum": np.cumsum(sr_pnl_steps_arr).tolist(),
            "noise_trader_pnl_cum": np.cumsum(noise_pnl_steps_arr).tolist(),
            "arb_pnl_cum": np.cumsum(arb_pnl_steps_arr).tolist(),
            # Keep smart-router routing metrics in light_mode so downstream dashboards can
            # build the DEX-share distribution without running the full visualization stack.
            "smart_router_dex_share_steps": list(sr_dex_share_steps),
            "smart_router_dex_share_series": list(sr_dex_share_series),
            "smart_router_dex_share_overall": float(sr_dex_share_overall),
            "smart_router_dex_share_mean": float(sr_dex_share_mean),
            "lp_pnl_active": list(lp_pnl_active_series),
            "lp_pnl_passive": list(lp_pnl_passive_series),
            "lp_unhedged_active": list(lp_unhedged_active_series),
            "lp_unhedged_passive": list(lp_unhedged_passive_series),
            "fee_series": list(fee_series),
            "total_noise_trader_swaps": int(total_noise_swaps_executed),
            "noise_trader_swaps_rejected_slippage": int(total_noise_swaps_skipped),
            "total_smart_router_swaps": int(total_smart_swaps_executed),
            "smart_router_swaps_rejected_slippage": int(total_smart_swaps_skipped),
            "smart_router_swaps_cex_routed": int(total_sr_cex_execs),
            "smart_router_swaps_dex_routed": int(total_sr_dex_execs),
            "total_arb_swaps": int(total_arb_swaps_executed),
            "arb_no_op_in_band": int(total_arb_no_op_in_band),
            "arb_swaps_rejected_profitability": int(total_arb_swaps_rejected_profitability),
            "total_jit_trades_executed": int(total_jit_trades_executed),
        }

    # =============================================================================
    # Plotting
    # =============================================================================
    P_series = np.array(P_series)
    M_series = np.array(M_series)
    X_active_end = np.array(X_active_end)
    Y_active_end = np.array(Y_active_end)
    band_lo_target = np.array(band_lo_target)
    band_hi_target = np.array(band_hi_target)
    L_end = np.array(L_end)
    L_pre_step = np.array(L_pre_step)
    L_pre_trader = np.array(L_pre_trader)
    L_pre_arb_eff = np.array(L_pre_arb_eff)
    steps = np.arange(len(P_series))
    trader_pnl_steps = np.array(trader_pnl_steps)
    arb_pnl_steps = np.array(arb_pnl_steps)
    trader_pnl_cum = np.cumsum(trader_pnl_steps)
    arb_pnl_cum = np.cumsum(arb_pnl_steps)
    sr_pnl_steps = np.array(sr_pnl_steps)
    noise_pnl_steps = np.array(noise_pnl_steps)
    sr_pnl_cum = np.cumsum(sr_pnl_steps)
    noise_pnl_cum = np.cumsum(noise_pnl_steps)
    sr_y_series = np.array(sr_y_series)
    noise_y_series = np.array(noise_y_series)
    dex_notional_y_series = np.array(dex_notional_y_series)
    lp_pnl_total_series = np.array(lp_pnl_total_series)
    lp_pnl_active_series = np.array(lp_pnl_active_series)
    lp_pnl_passive_series = np.array(lp_pnl_passive_series)
    lp_unhedged_total_series = np.array(lp_unhedged_total_series)
    lp_unhedged_active_series = np.array(lp_unhedged_active_series)
    lp_unhedged_passive_series = np.array(lp_unhedged_passive_series)
    lp_rebal_total_series = np.array(lp_rebal_total_series)
    lp_rebal_active_series = np.array(lp_rebal_active_series)
    lp_rebal_passive_series = np.array(lp_rebal_passive_series)
    lp_rebal_value_total_series = np.array(lp_rebal_value_total_series)
    lp_rebal_value_active_series = np.array(lp_rebal_value_active_series)
    lp_rebal_value_passive_series = np.array(lp_rebal_value_passive_series)
    lp_fee_value_total_series = np.array(lp_fee_value_total_series)
    lp_fee_value_active_series = np.array(lp_fee_value_active_series)
    lp_fee_value_passive_series = np.array(lp_fee_value_passive_series)
    lp_fees0_earned_total_series = np.array(lp_fees0_earned_total_series)
    lp_fees1_earned_total_series = np.array(lp_fees1_earned_total_series)
    lp_fees0_earned_active_series = np.array(lp_fees0_earned_active_series)
    lp_fees1_earned_active_series = np.array(lp_fees1_earned_active_series)
    lp_fees0_earned_passive_series = np.array(lp_fees0_earned_passive_series)
    lp_fees1_earned_passive_series = np.array(lp_fees1_earned_passive_series)
    lp_lvr_total_series = np.array(lp_lvr_total_series)
    lp_lvr_active_series = np.array(lp_lvr_active_series)
    lp_lvr_passive_series = np.array(lp_lvr_passive_series)
    lp_wallet_series = np.array(lp_wallet_series)
    lp_wallet_active_series = np.array(lp_wallet_active_series)
    lp_wallet_passive_series = np.array(lp_wallet_passive_series)
    lp_wealth_series = np.array(lp_wealth_series)
    lp_wealth_active_series = np.array(lp_wealth_active_series)
    lp_wealth_passive_series = np.array(lp_wealth_passive_series)
    jiter_wallet_series = np.array(jiter_wallet_series)
    jiter_wealth_series = np.array(jiter_wealth_series)
    jiter_fee_value_series = np.array(jiter_fee_value_series)
    jiter_fees0_earned_series = np.array(jiter_fees0_earned_series)
    jiter_fees1_earned_series = np.array(jiter_fees1_earned_series)
    jiter_position_value_series = np.array(jiter_position_value_series)
    jiter_pnl_series = np.array(jiter_pnl_series)
    jiter_flash_fee_paid_series = np.array(jiter_flash_fee_paid_series)
    fee_sigma_series = np.array(fee_sigma_series)
    fee_basis_ticks_series = np.array(fee_basis_ticks_series)
    fee_imb_series = np.array(fee_imb_series)
    fee_signal_series = np.array(fee_signal_series)
    cex_sigma_series = np.array(cex_sigma_series)
    cex_regime_series = np.array(cex_regime_series)
    sigma_panel = regime_mode or noisy_sine_mode or heston_mode

    # --- Agent activity series (per step, then cumulative) ---
    n_steps = len(P_series)
    smart_activity = np.zeros(n_steps, dtype=float)
    noise_activity = np.zeros(n_steps, dtype=float)
    lp_active_activity = np.zeros(n_steps, dtype=float)
    lp_passive_activity = np.zeros(n_steps, dtype=float)
    arb_activity = np.zeros(n_steps, dtype=float)
    jiter_activity = np.zeros(n_steps, dtype=float)

    # Smart router / noise trader: +1 for X->Y, -1 for Y->X
    for s, sign in zip(smart_activity_steps, smart_activity_signs):
        if 0 <= s < n_steps:
            smart_activity[s] += sign
    for s, sign in zip(noise_activity_steps, noise_activity_signs):
        if 0 <= s < n_steps:
            noise_activity[s] += sign

    # LPs (active vs passive): +1 for mint, -1 for burn
    for s, is_passive, is_jiter in zip(mint_steps, mint_is_passive, mint_is_jiter):
        if 0 <= s < n_steps:
            if is_jiter:
                continue
            target = lp_passive_activity if is_passive else lp_active_activity
            target[s] += 1.0
    for s, is_passive, is_jiter in zip(burn_steps, burn_is_passive, burn_is_jiter):
        if 0 <= s < n_steps:
            if is_jiter:
                continue
            target = lp_passive_activity if is_passive else lp_active_activity
            target[s] -= 1.0

    # Jiter: +1 mint, -1 burn
    for s, sign in zip(jiter_activity_steps, jiter_activity_signs):
        if 0 <= s < n_steps:
            jiter_activity[s] += sign

    # Arbitrageur: +1 for successful arb, 0 for skipped
    for s in arb_steps:
        if 0 <= s < n_steps:
            arb_activity[s] += 1.0

    smart_activity_cum = np.cumsum(smart_activity)
    noise_activity_cum = np.cumsum(noise_activity)
    lp_active_activity_cum = np.cumsum(lp_active_activity)
    lp_passive_activity_cum = np.cumsum(lp_passive_activity)
    arb_activity_cum = np.cumsum(arb_activity)
    jiter_activity_cum = np.cumsum(jiter_activity)

    if visualize:
        plotting_results(
            results_root=results_root_path,
            pid_str=pid_str,
            fee_mode=fee_mode,
            passive_lp_share=passive_lp_share,
            p_jit=p_jit,
            skip_step=int(skip_step),
            sigma_panel=sigma_panel,
            block_time=block_time,
            steps=steps,
            P_series=P_series,
            M_series=M_series,
            X_active_end=X_active_end,
            Y_active_end=Y_active_end,
            band_lo_target=band_lo_target,
            band_hi_target=band_hi_target,
            L_end=L_end,
            L_pre_step=L_pre_step,
            L_pre_trader=L_pre_trader,
            L_pre_arb_eff=L_pre_arb_eff,
            jiter_wealth_series=jiter_wealth_series,
            jiter_pnl_series=jiter_pnl_series,
            arb_pnl_cum=arb_pnl_cum,
            sr_pnl_cum=sr_pnl_cum,
            noise_pnl_cum=noise_pnl_cum,
            lp_pnl_active_series=lp_pnl_active_series,
            lp_pnl_passive_series=lp_pnl_passive_series,
            lp_unhedged_active_series=lp_unhedged_active_series,
            lp_unhedged_passive_series=lp_unhedged_passive_series,
            lp_fee_value_total_series=lp_fee_value_total_series,
            lp_lvr_total_series=lp_lvr_total_series,
            dex_notional_y_series=dex_notional_y_series,
            fee_series=fee_series,
            fee_sigma_series=fee_sigma_series,
            fee_basis_ticks_series=fee_basis_ticks_series,
            fee_signal_series=fee_signal_series,
            cex_sigma_series=cex_sigma_series,
            arb_y_series=arb_y_series,
            sr_y_series=sr_y_series,
            noise_y_series=noise_y_series,
            w_ticks_series=w_ticks_series,
            w_unclipped_series=w_unclipped_series,
            w_noise_series=w_noise_series,
            smart_activity_cum=smart_activity_cum,
            noise_activity_cum=noise_activity_cum,
            lp_active_activity_cum=lp_active_activity_cum,
            lp_passive_activity_cum=lp_passive_activity_cum,
            arb_activity_cum=arb_activity_cum,
            jiter_activity_cum=jiter_activity_cum,
            sr_dex_share_steps=sr_dex_share_steps,
            sr_dex_share_series=sr_dex_share_series,
            micro_steps=micro_steps,
            M_micro=M_micro,
            P_micro=P_micro,
            micro_valid_steps=micro_valid_steps,
            micro_valid_prices=micro_valid_prices,
            mint_steps=mint_steps,
            mint_sizes=mint_sizes,
            mint_is_passive=mint_is_passive,
            mint_is_jiter=mint_is_jiter,
            burn_steps=burn_steps,
            burn_sizes=burn_sizes,
            burn_is_passive=burn_is_passive,
            burn_is_jiter=burn_is_jiter,
            smart_router_enabled=smart_router_enabled,
            noise_trader_enabled=noise_trader_enabled,
            lp_active_enabled=lp_active_enabled,
            lp_passive_enabled=lp_passive_enabled,
            jiter_enabled=jiter_enabled,
        )

    return {
        "DEX_price": P_series,
        "CEX_price": M_series,
        "cex_sigma_series": cex_sigma_series.tolist(),
        "cex_regime_series": cex_regime_series.tolist(),
        "band_lo": band_lo_target,
        "band_hi": band_hi_target,
        "L_active_end": L_end,
        "L_pre_step": L_pre_step,
        "L_pre_trader": L_pre_trader,
        "L_pre_arb_eff": L_pre_arb_eff,
        "trader_notional_y": trader_y_series,
        "arb_notional_y": arb_y_series,
        "dex_notional_y": dex_notional_y_series.tolist(),
        "trader_steps": trader_steps,
        "trader_dirs": trader_dirs,
        "arb_steps": arb_steps,
        "arb_dirs": arb_dirs,
        "mint_steps": mint_steps,
        "mint_sizes": mint_sizes,
        "mint_widths": mint_widths,
        "burn_steps": burn_steps,
        "burn_sizes": burn_sizes,
        "liq_history": liq_history,
        "tick_history": tick_history,
        "x_active_reserves": X_active_end.tolist(),
        "y_active_reserves": Y_active_end.tolist(),
        "grid_base_s": pool.base_s,
        "grid_g": pool.g,
        "trader_pnl_steps": trader_pnl_steps.tolist(),
        "arb_pnl_steps": arb_pnl_steps.tolist(),
        "trader_pnl_cum": trader_pnl_cum.tolist(),
        "arb_pnl_cum": arb_pnl_cum.tolist(),
        "smart_router_pnl_steps": sr_pnl_steps.tolist(),
        "noise_trader_pnl_steps": noise_pnl_steps.tolist(),
        "smart_router_pnl_cum": sr_pnl_cum.tolist(),
        "noise_trader_pnl_cum": noise_pnl_cum.tolist(),
        "smart_router_notional_y": sr_y_series,
        "noise_trader_notional_y": noise_y_series,
        "smart_router_exec_count": sr_exec_count,
        "smart_router_cex_exec_count": sr_cex_exec_count,
        "smart_router_dex_exec_count": sr_dex_exec_count,
        "smart_router_dex_share_steps": sr_dex_share_steps,
        "smart_router_dex_share_series": sr_dex_share_series,
        "smart_router_dex_share_overall": sr_dex_share_overall,
        "smart_router_dex_share_mean": sr_dex_share_mean,
        "noise_trader_exec_count": noise_exec_count,
        "lp_pnl_total": lp_pnl_total_series.tolist(),
        "lp_pnl_active": lp_pnl_active_series.tolist(),
        "lp_pnl_passive": lp_pnl_passive_series.tolist(),
        "lp_unhedged_total": lp_unhedged_total_series.tolist(),
        "lp_unhedged_active": lp_unhedged_active_series.tolist(),
        "lp_unhedged_passive": lp_unhedged_passive_series.tolist(),
        "lp_rebal_total_series": lp_rebal_total_series.tolist(),
        "lp_rebal_active_series": lp_rebal_active_series.tolist(),
        "lp_rebal_passive_series": lp_rebal_passive_series.tolist(),
        "lp_rebal_value_total_series": lp_rebal_value_total_series.tolist(),
        "lp_rebal_value_active_series": lp_rebal_value_active_series.tolist(),
        "lp_rebal_value_passive_series": lp_rebal_value_passive_series.tolist(),
        "lp_fee_value_total_series": lp_fee_value_total_series.tolist(),
        "lp_fee_value_active_series": lp_fee_value_active_series.tolist(),
        "lp_fee_value_passive_series": lp_fee_value_passive_series.tolist(),
        "lp_fees0_earned_total_series": lp_fees0_earned_total_series.tolist(),
        "lp_fees1_earned_total_series": lp_fees1_earned_total_series.tolist(),
        "lp_fees0_earned_active_series": lp_fees0_earned_active_series.tolist(),
        "lp_fees1_earned_active_series": lp_fees1_earned_active_series.tolist(),
        "lp_fees0_earned_passive_series": lp_fees0_earned_passive_series.tolist(),
        "lp_fees1_earned_passive_series": lp_fees1_earned_passive_series.tolist(),
        "lp_lvr_total_series": lp_lvr_total_series.tolist(),
        "lp_lvr_active_series": lp_lvr_active_series.tolist(),
        "lp_lvr_passive_series": lp_lvr_passive_series.tolist(),
        "trader_exec_count": trader_exec_count,
        "fee_series": fee_series,
        "fee_mode": fee_mode,
        "f_min": f_min,
        "f_max": f_max,
        "fee_sigma_series": fee_sigma_series.tolist(),
        "fee_basis_ticks_series": fee_basis_ticks_series.tolist(),
        "fee_imb_series": fee_imb_series.tolist(),
        "fee_signal_series": fee_signal_series.tolist(),
        "lp_wallet_series": lp_wallet_series.tolist(),
        "lp_wealth_series": lp_wealth_series.tolist(),
        "lp_wallet_active_series": lp_wallet_active_series.tolist(),
        "lp_wallet_passive_series": lp_wallet_passive_series.tolist(),
        "lp_wealth_active_series": lp_wealth_active_series.tolist(),
        "lp_wealth_passive_series": lp_wealth_passive_series.tolist(),
        "jiter_wallet_series": jiter_wallet_series.tolist(),
        "jiter_wealth_series": jiter_wealth_series.tolist(),
        "jiter_fee_value_series": jiter_fee_value_series.tolist(),
        "jiter_fees0_earned_series": jiter_fees0_earned_series.tolist(),
        "jiter_fees1_earned_series": jiter_fees1_earned_series.tolist(),
        "jiter_position_value_series": jiter_position_value_series.tolist(),
        "jiter_pnl_series": jiter_pnl_series.tolist(),
        "jiter_flash_fee_paid_series": jiter_flash_fee_paid_series.tolist(),
        "jiter_activity_cum": jiter_activity_cum.tolist(),
        "arb_exec_count": arb_exec_count,
        "total_noise_trader_swaps": int(total_noise_swaps_executed),
        "noise_trader_swaps_rejected_slippage": int(total_noise_swaps_skipped),
        "total_smart_router_swaps": int(total_smart_swaps_executed),
        "smart_router_swaps_rejected_slippage": int(total_smart_swaps_skipped),
        "smart_router_swaps_cex_routed": int(total_sr_cex_execs),
        "smart_router_swaps_dex_routed": int(total_sr_dex_execs),
        "total_arb_swaps": int(total_arb_swaps_executed),
        "arb_no_op_in_band": int(total_arb_no_op_in_band),
        "arb_swaps_rejected_profitability": int(total_arb_swaps_rejected_profitability),
        "total_jit_trades_executed": int(total_jit_trades_executed),
        "smart_router_activity_cum": smart_activity_cum.tolist(),
        "noise_trader_activity_cum": noise_activity_cum.tolist(),
        "lp_active_activity_cum": lp_active_activity_cum.tolist(),
        "lp_passive_activity_cum": lp_passive_activity_cum.tolist(),
        "arb_activity_cum": arb_activity_cum.tolist(),
    }


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ABM Uni v3 simulation.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file containing a complete 'simulate' section.",
    )
    args = parser.parse_args()

    pid_str = str(os.getpid())
    print(f"[pid] {pid_str}")
    config_path = Path(args.config).expanduser().resolve()
    scenario_label, params = load_simulation_parameters(config_path, simulate_func=simulate)

    # Derive a scenario-specific output root under abm_results/scenarios/<name>
    scenario_root = scenario_output_root(config_path)
    params = dict(params)
    params["results_root"] = scenario_root

    print(f"[config] {config_path}")
    print(f"[scenario] {scenario_label} (output -> {scenario_root})")

    out = simulate(**params)

    # make liquidity GIF
    if bool(params.get("liquidity_for_gif", params.get("liquidty_for_gif", False))):
        make_liquidity_gif(
            liq_history=out["liq_history"],
            tick_history=out["tick_history"],
            base_s=out["grid_base_s"],
            g=out["grid_g"],
            out_path=str(
                scenario_root
                / f"liquidity_evolution_{scenario_label}_{params.get('cex_sigma')}_{params.get('T')}.gif"
            ),
            fps=20,
            dpi=120,
            pad_frac=0.05,
            downsample_every=10,
            center_line=True,
        )
