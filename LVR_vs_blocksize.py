#!/usr/bin/env python3
"""Sweep block size and compute per-block LVR-vs-fees metrics.

Goal
-----
Answer: *as the block size increases, how does the ratio between the LVR accrued in that
block and the fees earned in that block change?*

How it works
------------
1) Load a base scenario YAML (e.g., `abm_results/scenarios/test.yml`).
2) Build a grid of block sizes B (via `block_time`; default 2..16 inclusive).
3) For each B, run `--runs` simulations with different seeds (optionally in parallel via
   `--max-workers`).
4) Convert cumulative series produced by `run.simulate()` into *per-block increments* (after
   `--skip-step` burn-in), aggregate across runs, and plot the requested layers
   (default: medians only):
   - `--plot-medians`: per-B medians (pooled across runs/blocks)
   - `--plot-95-interval`: central 95% interval (2.5..97.5 percentiles; pooled across runs/blocks)
   - `--plot-means`: per-B means (pooled across runs/blocks)
   - `--plot-violin-plot`: per-B distributions (violin plots; pooled across runs/blocks)

Definitions (per block)
-----------------------
All fee values are expressed in token1 and evaluated at the *end-of-block* CEX price `m_t`.

Fees in the block:
  `ΔFees_block := (Δfees0_earned)*m_t + (Δfees1_earned)`

LVR accrued in the block (via hedged PnL deltas):
  - LP cohorts: `ΔLVR_block := ΔFees_block − ΔPnL_hedged`
  - Jiter:      `ΔLVR_block := ΔFees_block − (ΔPnL_hedged + ΔFlashPaid)`

Fee definition switch
---------------------
`--fee-definition` controls how `ΔFees_block` is computed:
  - `flow` (default): uses token-unit cumulative fee counters exported by `run.py` and values
    the *in-block* fee flow at `m_t` (no revaluation of previously earned token0 fees).
  - `mtm` (legacy): uses `np.diff(fees0_earned*m_t + fees1_earned)`, which includes a
    mark-to-market revaluation term on the existing token0 fee inventory.

Outputs
-------
Writes NPZ bundles and Plotly HTML/PNG files under:
  `abm_results/scenarios/<scenario_stem>/lvr_vs_blocksize/`

Example
-------
  conda activate main
  # Default: medians only
  python LVR_vs_blocksize.py --config abm_results/scenarios/test.yml --runs 50 --fee-definition flow

  # Overlay violin distributions and means
  python LVR_vs_blocksize.py --config abm_results/scenarios/test.yml --runs 50 --fee-definition flow \\
    --plot-violin-plot --plot-means --plot-medians
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import run as run_module
from utils import load_simulation_parameters, scenario_output_root


def _silent_tqdm(iterable=None, **kwargs):
    """Silence tqdm inside run.simulate to avoid nested progress bars.

    Parameters:
        iterable: Optional iterable to wrap.
        **kwargs: Ignored tqdm options.

    Returns:
        Iterable: A pass-through iterable.

    Notes:
        `run.py` uses `tqdm(range(...))`. Replacing it with `range(...)` prevents
        worker subprocesses from generating progress bars.
    """
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


# Monkeypatch tqdm used inside run.py
run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]

simulate = run_module.simulate


@dataclass(frozen=True)
class CohortSpec:
    """Metadata for a cohort whose ΔLVR distribution we want to plot."""

    name: str
    label: str
    color: str


COHORT_SPECS: Tuple[CohortSpec, ...] = (
    CohortSpec("active", "Active LPs", "#9467bd"),
    CohortSpec("passive", "Passive LPs", "#8c564b"),
    CohortSpec("jiter", "Jiter", "#d62728"),
)


def _make_unique_dir(path: Path) -> Path:
    """Create a unique directory, appending a suffix if needed.

    Parameters:
        path: Target directory path to create.

    Returns:
        Path: The created directory path (may differ from the input if a suffix
        was added).

    Notes:
        If `path` already exists, a suffix `_<n>` is appended until an available
        path is found. This avoids collisions across concurrent invocations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.name}_{suffix}")
            suffix += 1


def _resolve_enabled_cohorts(params: Mapping[str, Any]) -> List[CohortSpec]:
    """Determine which cohorts are active given the scenario parameters.

    Parameters:
        params: Simulation parameter mapping (loaded from YAML + defaults).

    Returns:
        List[CohortSpec]: Enabled cohorts, in a stable order.

    Notes:
        - Active LP cohort exists iff `passive_lp_share < 1`.
        - Passive LP cohort exists iff `passive_lp_share > 0`.
        - Jiter exists iff `p_jit > 0`, `N_jit > 0`, and `liquidity_perc_jit > 0`.
    """
    try:
        passive_share = float(params.get("passive_lp_share", 1.0))
    except (TypeError, ValueError):
        passive_share = 1.0
    passive_share = max(0.0, min(1.0, passive_share))

    try:
        p_jit = float(params.get("p_jit", 0.0))
    except (TypeError, ValueError):
        p_jit = 0.0
    try:
        n_jit = int(params.get("N_jit", 0))
    except (TypeError, ValueError):
        n_jit = 0
    try:
        liq_perc = float(params.get("liquidity_perc_jit", 0.0))
    except (TypeError, ValueError):
        liq_perc = 0.0

    include_active = passive_share < 1.0
    include_passive = passive_share > 0.0
    include_jiter = p_jit > 0.0 and n_jit > 0 and liq_perc > 0.0

    enabled: List[CohortSpec] = []
    for spec in COHORT_SPECS:
        if spec.name == "active" and not include_active:
            continue
        if spec.name == "passive" and not include_passive:
            continue
        if spec.name == "jiter" and not include_jiter:
            continue
        enabled.append(spec)
    return enabled


def _delta_after_skip(values: Sequence[float], *, skip_step: int) -> np.ndarray:
    """Compute per-block increments Δx after skipping a transient prefix.

    Parameters:
        values: Cumulative series x_t over blocks (length ~= T).
        skip_step: Number of initial blocks to omit (burn-in).

    Returns:
        np.ndarray: Δx series after burn-in (float), with non-finite values set to NaN.

    Notes:
        The Δ series is computed as `diff(values[skip_step:])`, which excludes
        the increment that bridges the transient regime boundary.
    """
    arr = np.asarray(values, dtype=float)
    s0 = max(0, min(int(skip_step), int(arr.size)))
    arr = arr[s0:]
    if arr.size < 2:
        return np.array([], dtype=float)
    delta = np.diff(arr)
    delta[~np.isfinite(delta)] = np.nan
    return delta


def _ratio_after_skip(
    numer_cum: Sequence[float],
    denom_cum: Sequence[float],
    *,
    skip_step: int,
    eps: float = 1e-18,
) -> np.ndarray:
    """Compute per-block ratio Δnumer / Δdenom after skipping a transient prefix.

    Parameters:
        numer_cum: Cumulative numerator series (e.g., LVR_t) over blocks (length ~= T).
        denom_cum: Cumulative denominator series (e.g., fee_value_t) over blocks (length ~= T).
        skip_step: Number of initial blocks to omit (burn-in).
        eps: Small positive threshold to treat denominators as non-positive.

    Returns:
        np.ndarray: Ratio series after burn-in (float). Values are NaN when Δdenom <= eps
        or when non-finite values are encountered.

    Notes:
        This is used to compute the "fee coverage" ratio:
            ΔLVR_t / ΔFees_t
        where both LVR and Fees are measured in token1 value (marked-to-market).
        We intentionally exclude blocks with non-positive ΔFees because fee value can
        decrease when the numeraire price moves, which makes the ratio ambiguous.
    """
    a = np.asarray(numer_cum, dtype=float)
    b = np.asarray(denom_cum, dtype=float)
    n = min(int(a.size), int(b.size))
    if n <= 0:
        return np.array([], dtype=float)
    a = a[:n]
    b = b[:n]

    s0 = max(0, min(int(skip_step), int(n)))
    a = a[s0:]
    b = b[s0:]
    if a.size < 2 or b.size < 2:
        return np.array([], dtype=float)

    da = np.diff(a)
    db = np.diff(b)
    ratio = np.full_like(da, np.nan, dtype=float)
    mask = np.isfinite(da) & np.isfinite(db) & (db > float(eps))
    ratio[mask] = da[mask] / db[mask]
    return ratio


def _fee_flow_value_after_skip(
    fees0_cum: Sequence[float],
    fees1_cum: Sequence[float],
    m_series: Sequence[float],
    *,
    skip_step: int,
) -> np.ndarray:
    """Compute per-block fee *flow value* ΔFees_block after skipping burn-in.

    Parameters:
        fees0_cum: Cumulative token0 fee counter `fees0_earned` (token0 units) per block.
        fees1_cum: Cumulative token1 fee counter `fees1_earned` (token1 units) per block.
        m_series: End-of-block CEX price series m_t (token1 per token0) per block.
        skip_step: Number of initial blocks to omit (burn-in).

    Returns:
        np.ndarray: Per-block fee flow value series (token1 units), defined as:
            ΔFees_block[t] = (fees0_cum[t] - fees0_cum[t-1]) * m_t
                           + (fees1_cum[t] - fees1_cum[t-1]).

        The returned series is aligned with other Δ-series computed via `np.diff(...)`
        after burn-in (length ~= T - skip_step - 1). Non-finite values, or negative
        fee-counter deltas, are set to NaN.

    Notes:
        This differs from `np.diff(fees0_cum*m_series + fees1_cum)` which includes a
        revaluation term `fees0_cum[t-1] * (m_t - m_{t-1})`. We intentionally exclude
        that term so the denominator represents fees earned *in the block*, valued at
        the end-of-block CEX price.
    """
    f0 = np.asarray(fees0_cum, dtype=float)
    f1 = np.asarray(fees1_cum, dtype=float)
    m = np.asarray(m_series, dtype=float)
    n = min(int(f0.size), int(f1.size), int(m.size))
    if n <= 0:
        return np.array([], dtype=float)
    f0 = f0[:n]
    f1 = f1[:n]
    m = m[:n]

    s0 = max(0, min(int(skip_step), int(n)))
    f0 = f0[s0:]
    f1 = f1[s0:]
    m = m[s0:]
    if f0.size < 2 or f1.size < 2 or m.size < 2:
        return np.array([], dtype=float)

    df0 = np.diff(f0)
    df1 = np.diff(f1)
    m_t = m[1:]  # value at end-of-block t for the corresponding Δ

    dfees = df0 * m_t + df1
    bad = (~np.isfinite(dfees)) | (~np.isfinite(df0)) | (~np.isfinite(df1)) | (~np.isfinite(m_t))
    bad |= (df0 < 0.0) | (df1 < 0.0)
    dfees = np.asarray(dfees, dtype=float)
    dfees[bad] = np.nan
    return dfees


def _lvr_block_and_ratio_from_fee_flow_after_skip(
    fees0_cum: Sequence[float],
    fees1_cum: Sequence[float],
    pnl_series: Sequence[float],
    m_series: Sequence[float],
    *,
    skip_step: int,
    extra_cost_cum: Optional[Sequence[float]] = None,
    eps: float = 1e-18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-block ΔLVR_block and ΔLVR_block/ΔFees_block after burn-in.

    Parameters:
        fees0_cum: Cumulative token0 fee counter `fees0_earned` (token0 units) per block.
        fees1_cum: Cumulative token1 fee counter `fees1_earned` (token1 units) per block.
        pnl_series: Hedged PnL series (token1 units) per block.
        m_series: End-of-block CEX price series m_t (token1 per token0) per block.
        skip_step: Number of initial blocks to omit (burn-in).
        extra_cost_cum: Optional cumulative external cost series to subtract (token1 units).
            This is used for the Jiter to remove flash-loan financing costs:
                ΔLVR_block = ΔFees_block - (ΔPnL + ΔFlashPaid).
            For regular LP cohorts, keep this as None.
        eps: Small positive threshold used to mask near-zero denominators.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - ΔLVR_block: Per-block LVR increments computed from fee flow and hedged PnL deltas.
            - ratio: ΔLVR_block / ΔFees_block, with NaN where ΔFees_block <= eps or non-finite.

    Notes:
        Definitions (per block t, after burn-in):
            ΔFees_block[t] = (Δfees0_earned[t]) * m_t + (Δfees1_earned[t])
            ΔPnL[t]        = pnl[t] - pnl[t-1]
            ΔLVR_block[t]  = ΔFees_block[t] - ΔPnL[t] - ΔExtraCost[t]   (if provided)

        This is designed to answer: "as block size increases, how does the ratio of
        LVR accrued in the block to fees earned in the block change?" without mixing
        in mark-to-market revaluation of previously earned token0 fees.
    """
    f0 = np.asarray(fees0_cum, dtype=float)
    f1 = np.asarray(fees1_cum, dtype=float)
    pnl = np.asarray(pnl_series, dtype=float)
    m = np.asarray(m_series, dtype=float)
    n = min(int(f0.size), int(f1.size), int(pnl.size), int(m.size))
    extra: Optional[np.ndarray] = None
    if extra_cost_cum is not None:
        extra = np.asarray(extra_cost_cum, dtype=float)
        n = min(n, int(extra.size))

    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    f0 = f0[:n]
    f1 = f1[:n]
    pnl = pnl[:n]
    m = m[:n]
    if extra is not None:
        extra = extra[:n]

    s0 = max(0, min(int(skip_step), int(n)))
    f0 = f0[s0:]
    f1 = f1[s0:]
    pnl = pnl[s0:]
    m = m[s0:]
    if extra is not None:
        extra = extra[s0:]

    if f0.size < 2 or f1.size < 2 or pnl.size < 2 or m.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    dfees = _fee_flow_value_after_skip(f0, f1, m, skip_step=0)  # already sliced
    dpnl = np.diff(pnl)
    dcost_arr = np.zeros_like(dpnl, dtype=float) if extra is None else np.diff(extra)

    # Align lengths defensively (should be identical).
    k = min(int(dfees.size), int(dpnl.size), int(dcost_arr.size))
    if k <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    dfees = dfees[:k]
    dpnl = np.asarray(dpnl, dtype=float)[:k]
    dcost_arr = np.asarray(dcost_arr, dtype=float)[:k]

    dlvr = dfees - (dpnl + dcost_arr)
    dlvr[~np.isfinite(dlvr)] = np.nan

    ratio = np.full_like(dlvr, np.nan, dtype=float)
    mask = np.isfinite(dlvr) & np.isfinite(dfees) & (dfees > float(eps))
    ratio[mask] = dlvr[mask] / dfees[mask]
    return dlvr, ratio


def _compute_jiter_lvr_cumulative(out: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct Jiter cumulative LVR from returned accounting series.

    Parameters:
        out: Output dict returned by `run.simulate()`.

    Returns:
        np.ndarray: Cumulative Jiter LVR series (float).

    Notes:
        In the implementation, `jiter_pnl_series` is the hedged PnL measured as
        V^LP - V^reb, and flash-loan fees are debited directly from the wallet.
        The model's identity implies:

            jiter_pnl = (F - LVR) - flash_fees_paid
            => LVR = F - (jiter_pnl + flash_fees_paid)
    """
    fee = np.asarray(out.get("jiter_fee_value_series", []), dtype=float)
    pnl = np.asarray(out.get("jiter_pnl_series", []), dtype=float)
    flash = np.asarray(out.get("jiter_flash_fee_paid_series", []), dtype=float)

    n = min(int(fee.size), int(pnl.size), int(flash.size))
    if n <= 0:
        return np.array([], dtype=float)

    fee = fee[:n]
    pnl = pnl[:n]
    flash = flash[:n]
    return fee - (pnl + flash)


def _jit_success_mask_after_skip(out: Mapping[str, Any], *, skip_step: int, n_steps: int) -> np.ndarray:
    """Build a boolean mask for blocks with successful JIT execution after burn-in.

    Parameters:
        out: Output dict returned by `run.simulate()`.
        skip_step: Number of initial blocks to omit (burn-in).
        n_steps: Number of simulation blocks to align to (typically len(LVR series)).

    Returns:
        np.ndarray: Boolean mask aligned with Δ series (length ~= n_steps - skip_step - 1).

    Notes:
        `run.py` increments `jiter_activity` by +1 in block `t` when a JIT mint/burn
        roundtrip successfully surrounds an executed swap (`jit_swap_executed=True`).
        The output only exposes the cumulative series `jiter_activity_cum`, so we
        recover per-block counts via a first difference.
    """
    cum = np.asarray(out.get("jiter_activity_cum", []), dtype=float)
    if cum.size <= 0 or n_steps <= 0:
        return np.array([], dtype=bool)
    n = min(int(n_steps), int(cum.size))
    cum = cum[:n]
    # Per-block successful jit count (0,1,2,...) via diff of cumulative series.
    act = np.diff(np.concatenate(([0.0], cum)))

    s0 = max(0, min(int(skip_step), n))
    # Align mask with ΔLVR = LVR[t] - LVR[t-1] for t=s0+1..n-1
    mask = act[s0 + 1 :]
    return mask > 0.0


def _run_one(
    seed: int,
    block_time: int,
    *,
    base_params: Mapping[str, Any],
    tmp_root: Path,
    keep_run_artifacts: bool,
    cohort_names: Sequence[str],
    fee_definition: str,
) -> Dict[str, Any]:
    """Worker: run one simulation and return ΔLVR and ΔLVR/ΔFees series per enabled cohort.

    Parameters:
        seed: RNG seed for the simulation.
        block_time: Block size B (micro-steps per block).
        base_params: Base simulation parameters (will be copied and overridden).
        tmp_root: Temporary output root for per-run artifacts (logs, etc.).
        keep_run_artifacts: If True, keep per-run temp folders. Otherwise delete them.
        cohort_names: Cohort names to return (subset of {"active","passive","jiter"}).
        fee_definition: Which fee definition to use in the ratio computation:
            - "flow": per-block fee flow value ΔFees_block = Δfees0_earned*m_t + Δfees1_earned.
            - "mtm": mark-to-market ΔFeeValue = Δ(fees0_earned*m_t + fees1_earned) (legacy).

    Returns:
        Dict[str, Any]: A payload with keys: block_time, seed, `dLVR_<cohort>`,
        and `dLVR_over_dFees_<cohort>`.

    Notes:
        We call `run.simulate()` directly to keep results in-memory. We set
        `results_root` to a per-run folder so any verbose logs are isolated and
        can be removed when `keep_run_artifacts=False`.
    """
    import run as _run_module

    _run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]

    params = dict(base_params)
    params["seed"] = int(seed)
    params["block_time"] = int(block_time)

    run_dir = tmp_root / f"B{int(block_time)}_seed{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params["results_root"] = run_dir

    out = _run_module.simulate(**params)
    skip_step = int(params.get("skip_step", 0))

    payload: Dict[str, Any] = {
        "seed": int(seed),
        "block_time": int(block_time),
    }

    if "active" in cohort_names:
        if str(fee_definition) == "flow":
            fees0_cum = out.get("lp_fees0_earned_active_series", [])
            fees1_cum = out.get("lp_fees1_earned_active_series", [])
            pnl = out.get("lp_pnl_active", [])
            m_series = out.get("CEX_price", [])
            dlvr, ratio = _lvr_block_and_ratio_from_fee_flow_after_skip(
                fees0_cum,
                fees1_cum,
                pnl,
                m_series,
                skip_step=skip_step,
            )
            payload["dLVR_active"] = dlvr
            payload["dLVR_over_dFees_active"] = ratio
        else:
            lvr_cum = out.get("lp_lvr_active_series", [])
            fee_cum = out.get("lp_fee_value_active_series", [])
            payload["dLVR_active"] = _delta_after_skip(lvr_cum, skip_step=skip_step)
            payload["dLVR_over_dFees_active"] = _ratio_after_skip(lvr_cum, fee_cum, skip_step=skip_step)
    if "passive" in cohort_names:
        if str(fee_definition) == "flow":
            fees0_cum = out.get("lp_fees0_earned_passive_series", [])
            fees1_cum = out.get("lp_fees1_earned_passive_series", [])
            pnl = out.get("lp_pnl_passive", [])
            m_series = out.get("CEX_price", [])
            dlvr, ratio = _lvr_block_and_ratio_from_fee_flow_after_skip(
                fees0_cum,
                fees1_cum,
                pnl,
                m_series,
                skip_step=skip_step,
            )
            payload["dLVR_passive"] = dlvr
            payload["dLVR_over_dFees_passive"] = ratio
        else:
            lvr_cum = out.get("lp_lvr_passive_series", [])
            fee_cum = out.get("lp_fee_value_passive_series", [])
            payload["dLVR_passive"] = _delta_after_skip(lvr_cum, skip_step=skip_step)
            payload["dLVR_over_dFees_passive"] = _ratio_after_skip(lvr_cum, fee_cum, skip_step=skip_step)
    if "jiter" in cohort_names:
        if str(fee_definition) == "flow":
            fees0_cum = out.get("jiter_fees0_earned_series", [])
            fees1_cum = out.get("jiter_fees1_earned_series", [])
            pnl = out.get("jiter_pnl_series", [])
            m_series = out.get("CEX_price", [])
            flash_cum = out.get("jiter_flash_fee_paid_series", [])
            dlvr, ratio = _lvr_block_and_ratio_from_fee_flow_after_skip(
                fees0_cum,
                fees1_cum,
                pnl,
                m_series,
                skip_step=skip_step,
                extra_cost_cum=flash_cum,
            )
            payload["dLVR_jiter"] = dlvr
            payload["dLVR_over_dFees_jiter"] = ratio

            act_cum = out.get("jiter_activity_cum", [])
            n_steps = min(
                int(np.asarray(fees0_cum, dtype=float).size),
                int(np.asarray(fees1_cum, dtype=float).size),
                int(np.asarray(pnl, dtype=float).size),
                int(np.asarray(m_series, dtype=float).size),
                int(np.asarray(flash_cum, dtype=float).size),
                int(np.asarray(act_cum, dtype=float).size),
            )
            payload["jit_success_mask"] = _jit_success_mask_after_skip(out, skip_step=skip_step, n_steps=n_steps)
        else:
            jiter_lvr_cum = _compute_jiter_lvr_cumulative(out)
            jiter_fee_cum = np.asarray(out.get("jiter_fee_value_series", []), dtype=float)
            # Align to the activity series length so the success mask matches ΔLVR indexing.
            act_cum = np.asarray(out.get("jiter_activity_cum", []), dtype=float)
            if jiter_lvr_cum.size > 0:
                n = int(jiter_lvr_cum.size)
                if act_cum.size > 0:
                    n = min(n, int(act_cum.size))
                if jiter_fee_cum.size > 0:
                    n = min(n, int(jiter_fee_cum.size))
                jiter_lvr_cum = jiter_lvr_cum[:n]
                jiter_fee_cum = jiter_fee_cum[:n]
            # Keep unconditional per-block ΔLVR for storage; apply conditional filtering
            # (successful JIT blocks) downstream when computing medians/distributions.
            payload["dLVR_jiter"] = _delta_after_skip(jiter_lvr_cum, skip_step=skip_step)
            payload["dLVR_over_dFees_jiter"] = _ratio_after_skip(jiter_lvr_cum, jiter_fee_cum, skip_step=skip_step)
            payload["jit_success_mask"] = _jit_success_mask_after_skip(
                out, skip_step=skip_step, n_steps=int(jiter_lvr_cum.size)
            )

    if not keep_run_artifacts:
        shutil.rmtree(run_dir, ignore_errors=True)

    return payload


def _build_violin_figure(
    distributions: Mapping[str, Mapping[int, np.ndarray]],
    *,
    cohort_specs: Sequence[CohortSpec],
    block_times: Sequence[int],
    yaxis_title: str = "ΔLVR",
    plot_medians: bool = True,
    plot_95_interval: bool = False,
    plot_means: bool = False,
    width_per_col: int = 520,
    height: int = 520,
    # title: str,
) -> go.Figure:
    """Create violin plots of per-block distributions vs block size.

    Parameters:
        distributions: Mapping cohort -> block_time -> 1D samples for the plotted metric.
        cohort_specs: Enabled cohort specifications (label/color).
        block_times: Block size grid in display order.
        yaxis_title: Y-axis label (metric name).
        plot_medians: If True, overlay a median line (computed from pooled samples).
        plot_95_interval: If True and `plot_medians` is True, overlay a central 95% interval
            (2.5th..97.5th percentiles) around the median points.
        plot_means: If True, overlay a mean line (computed from pooled samples).
        width_per_col: Plot width per cohort column (pixels).
        height: Plot height (pixels).

    Returns:
        go.Figure: Plotly figure with one column per cohort.

    Notes:
        Each violin is built from the pooled samples across all runs for a
        given block size. Medians/means are computed on the pooled samples.
    """
    n_cols = max(1, len(cohort_specs))
    fig_width = int(max(900, width_per_col * n_cols))
    subplot_titles = [spec.label for spec in cohort_specs] if cohort_specs else [yaxis_title]
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    x_categories = [str(int(b)) for b in block_times]
    show_legend = bool(plot_means)

    for col_idx, spec in enumerate(cohort_specs, start=1):
        per_b = distributions.get(spec.name, {})

        medians: List[float] = []
        p2_5s: List[float] = []
        p97_5s: List[float] = []
        means: List[float] = []
        for b in block_times:
            vals = np.asarray(per_b.get(int(b), np.array([], dtype=float)), dtype=float)
            vals = vals[np.isfinite(vals)]
            med = float(np.median(vals)) if vals.size > 0 else float("nan")
            medians.append(med)
            if vals.size > 0:
                p2_5, p97_5 = [float(x) for x in np.percentile(vals, [2.5, 97.5])]
            else:
                p2_5, p97_5 = float("nan"), float("nan")
            p2_5s.append(p2_5)
            p97_5s.append(p97_5)
            means.append(float(np.mean(vals)) if vals.size > 0 else float("nan"))

            fig.add_trace(
                go.Violin(
                    x=[str(int(b))] * int(vals.size),
                    y=vals,
                    name=str(int(b)),
                    showlegend=False,
                    points=False,
                    line=dict(color=spec.color, width=1.2),
                    fillcolor=spec.color,
                    opacity=0.35,
                    meanline_visible=False,
                    scalemode="width",
                ),
                row=1,
                col=col_idx,
            )

        if bool(plot_medians):
            error_y = None
            customdata = None
            hovertemplate = "B=%{x}<br>median=%{y:.6g}<extra></extra>"
            if bool(plot_95_interval):
                err_plus: List[Optional[float]] = []
                err_minus: List[Optional[float]] = []
                customdata = []
                for med, lo, hi in zip(medians, p2_5s, p97_5s):
                    if np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi):
                        err_plus.append(float(hi - med))
                        err_minus.append(float(med - lo))
                    else:
                        err_plus.append(None)
                        err_minus.append(None)
                    customdata.append([float(lo), float(hi)])
                error_y = dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus)
                hovertemplate = (
                    "B=%{x}<br>"
                    "median=%{y:.6g}<br>"
                    "p2.5=%{customdata[0]:.6g}<br>"
                    "p97.5=%{customdata[1]:.6g}"
                    "<extra></extra>"
                )
            fig.add_trace(
                go.Scatter(
                    x=x_categories,
                    y=medians,
                    mode="lines+markers",
                    name="Median",
                    showlegend=show_legend and col_idx == 1,
                    legendgroup="median",
                    line=dict(color="black", width=2),
                    marker=dict(color="black", size=6),
                    error_y=error_y,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=col_idx,
            )

        if bool(plot_means):
            fig.add_trace(
                go.Scatter(
                    x=x_categories,
                    y=means,
                    mode="lines+markers",
                    name="Mean",
                    showlegend=show_legend and col_idx == 1,
                    legendgroup="mean",
                    line=dict(color="#D55E00", width=2, dash="dot"),
                    marker=dict(color="#D55E00", size=7, symbol="x"),
                    hovertemplate="B=%{x}<br>mean=%{y:.6g}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

        fig.add_hline(
            y=0.0,
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=col_idx,
        )
        if col_idx == 1:
            fig.update_yaxes(title_text=yaxis_title, row=1, col=col_idx)

    fig.update_xaxes(title_text="B", categoryorder="array", categoryarray=x_categories)
    fig.update_layout(
        template="plotly_white",
        # title=title,
        violinmode="group",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        font=dict(size=18, color="black"),
        margin=dict(t=120, b=80),
        width=fig_width,
        height=int(height),
    )
    return fig


def _build_medians_only_figure(
    medians: Mapping[str, Sequence[float]],
    *,
    means: Optional[Mapping[str, Sequence[float]]] = None,
    p2_5: Optional[Mapping[str, Sequence[float]]] = None,
    p97_5: Optional[Mapping[str, Sequence[float]]] = None,
    cohort_specs: Sequence[CohortSpec],
    block_times: Sequence[int],
    yaxis_title: str = "ΔLVR",
    plot_medians: bool = True,
    plot_95_interval: bool = False,
    plot_means: bool = False,
    width_per_col: int = 520,
    height: int = 460,
    # title: str,
) -> go.Figure:
    """Create a median/mean plot vs block size (no violins).

    Parameters:
        medians: Mapping cohort -> sequence of median values aligned with `block_times`.
        means: Optional mapping cohort -> sequence of mean values aligned with `block_times`.
        p2_5: Optional mapping cohort -> sequence of 2.5th percentile values aligned with `block_times`.
        p97_5: Optional mapping cohort -> sequence of 97.5th percentile values aligned with `block_times`.
        cohort_specs: Enabled cohort specifications (label/color).
        block_times: Block size grid in display order.
        yaxis_title: Y-axis label (metric name).
        plot_medians: If True, plot the median line.
        plot_95_interval: If True and `plot_medians` is True, plot the central 95% interval
            (2.5th..97.5th percentiles) around the median points.
        plot_means: If True, plot the mean line.
        width_per_col: Plot width per cohort column (pixels).
        height: Plot height (pixels).

    Returns:
        go.Figure: Plotly figure with one column per cohort.

    Notes:
        The plotted median is the median of pooled samples across all runs (and all
        post-transient blocks) for each block size B.
    """
    n_cols = max(1, len(cohort_specs))
    fig_width = int(max(900, width_per_col * n_cols))
    subplot_titles = [spec.label for spec in cohort_specs] if cohort_specs else [yaxis_title]
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    x_categories = [str(int(b)) for b in block_times]
    show_legend = bool(plot_means)

    for col_idx, spec in enumerate(cohort_specs, start=1):
        y = list(medians.get(spec.name, []))
        if len(y) != len(x_categories):
            y = [float("nan")] * len(x_categories)

        if bool(plot_medians):
            error_y = None
            customdata = None
            hovertemplate = "B=%{x}<br>median=%{y:.6g}<extra></extra>"
            if bool(plot_95_interval) and p2_5 is not None and p97_5 is not None:
                y_lo = list(p2_5.get(spec.name, []))
                y_hi = list(p97_5.get(spec.name, []))
                if len(y_lo) == len(x_categories) and len(y_hi) == len(x_categories):
                    err_plus: List[Optional[float]] = []
                    err_minus: List[Optional[float]] = []
                    customdata = []
                    for med, lo, hi in zip(y, y_lo, y_hi):
                        if np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi):
                            err_plus.append(float(hi - med))
                            err_minus.append(float(med - lo))
                        else:
                            err_plus.append(None)
                            err_minus.append(None)
                        customdata.append([float(lo), float(hi)])
                    error_y = dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus)
                    hovertemplate = (
                        "B=%{x}<br>"
                        "median=%{y:.6g}<br>"
                        "p2.5=%{customdata[0]:.6g}<br>"
                        "p97.5=%{customdata[1]:.6g}"
                        "<extra></extra>"
                    )
            fig.add_trace(
                go.Scatter(
                    x=x_categories,
                    y=y,
                    mode="lines+markers",
                    name="Median",
                    showlegend=show_legend and col_idx == 1,
                    legendgroup="median",
                    line=dict(color="black", width=2),
                    marker=dict(color="black", size=6),
                    error_y=error_y,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=col_idx,
            )

        if bool(plot_means):
            y_mean = list((means or {}).get(spec.name, []))
            if len(y_mean) != len(x_categories):
                y_mean = [float("nan")] * len(x_categories)
            fig.add_trace(
                go.Scatter(
                    x=x_categories,
                    y=y_mean,
                    mode="lines+markers",
                    name="Mean",
                    showlegend=show_legend and col_idx == 1,
                    legendgroup="mean",
                    line=dict(color="#D55E00", width=2, dash="dot"),
                    marker=dict(color="#D55E00", size=7, symbol="x"),
                    hovertemplate="B=%{x}<br>mean=%{y:.6g}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

        fig.add_hline(
            y=0.0,
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=col_idx,
        )
        if col_idx == 1:
            fig.update_yaxes(title_text=yaxis_title, row=1, col=col_idx)

    fig.update_xaxes(title_text="B", categoryorder="array", categoryarray=x_categories)
    fig.update_layout(
        template="plotly_white",
        # title=title,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        font=dict(size=18, color="black"),
        margin=dict(t=120, b=80),
        width=fig_width,
        height=int(height),
    )
    return fig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters:
        None.

    Returns:
        argparse.Namespace: Parsed CLI options.

    Notes:
        Defaults are chosen to match the requested experiment:
        B=2..16 (15 points), N_run=10, and seed base from the YAML when omitted.

    Examples:
        >>> args = parse_args()
    """
    p = argparse.ArgumentParser(description="Sweep block_time and plot per-block ΔLVR metrics vs block size.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("abm_results/scenarios/test.yml"),
        help="Path to the YAML scenario config. Default: abm_results/scenarios/test.yml",
    )
    p.add_argument("--runs", type=int, default=10, help="Number of runs/seeds per block size. Default: 10.")
    p.add_argument("--block-min", type=int, default=2, help="Minimum block_time (inclusive). Default: 2.")
    p.add_argument("--block-max", type=int, default=16, help="Maximum block_time (inclusive). Default: 16.")
    p.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Base seed (defaults to the YAML seed). Seeds are seed_base + i*seed_step.",
    )
    p.add_argument("--seed-step", type=int, default=1, help="Seed increment per run. Default: 1.")
    p.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel worker processes. Default: CPUs minus one.",
    )
    p.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        help="Keep per-run temp folders/logs. Default is to delete them.",
    )
    p.add_argument(
        "--load-npz",
        type=Path,
        default=None,
        help=(
            "Load a previously saved `dLVR_arrays_*.npz` and regenerate plots/CSVs without re-running simulations. "
            "When provided, --runs/--block-min/--block-max/--seed-* are ignored."
        ),
    )
    p.add_argument(
        "--plot-medians",
        "--plot_medians",
        action="store_true",
        help="Plot per-B medians (line). Enabled by default; flag kept for explicitness.",
    )
    p.add_argument(
        "--plot-95-interval",
        "--plot_95_interval",
        action="store_true",
        help=(
            "When plotting medians, overlay the central 95% interval of the pooled samples "
            "(2.5th..97.5th percentiles) as error bars. Ignored if medians are not plotted."
        ),
    )
    p.add_argument(
        "--plot-means",
        "--plot_means",
        action="store_true",
        help="Plot per-B means (line). Can be combined with --plot-medians and/or --plot-violin-plot.",
    )
    p.add_argument(
        "--plot-violin-plot",
        "--plot_violin_plot",
        action="store_true",
        help=(
            "Plot per-B distributions as violin plots (pooled across runs). "
            "Can be combined with --plot-medians and/or --plot-means."
        ),
    )
    p.add_argument(
        "--plot-only-medians",
        "--plot_only_medians",
        action="store_true",
        help="[DEPRECATED] Medians are the default; kept for compatibility.",
    )
    p.add_argument(
        "--plot-medians-mean",
        "--plot_medians_mean",
        action="store_true",
        help="[DEPRECATED] Equivalent to --plot-medians --plot-means; kept for compatibility.",
    )
    p.add_argument(
        "--fee-definition",
        choices=("flow", "mtm"),
        default="flow",
        help=(
            "Fee definition used for the per-block ΔLVR and ΔLVR/ΔFees computation. "
            "'flow' uses ΔFees_block = Δfees0_earned*m_t + Δfees1_earned (end-of-block valuation; no revaluation of old fees). "
            "'mtm' uses the legacy mark-to-market Δ(fees0_earned*m_t + fees1_earned). Default: flow."
        ),
    )
    return p.parse_args()


def _infer_label_stub_from_npz_path(npz_path: Path) -> str:
    """Infer the label stub used in output filenames from an NPZ filename.

    Parameters:
        npz_path: Path to a `dLVR_arrays_*.npz` file produced by this script.

    Returns:
        str: The inferred label stub (typically `<fee_mode>_<scenario_stem>`). Falls back
        to `npz_path.stem` when the expected pattern is not found.

    Notes:
        The script saves NPZ files as:
            dLVR_arrays_<label_stub>_runsN_B..._pidXXXX.npz
        where `<label_stub>` is reused by the derived CSV/HTML/PNG filenames. When loading
        arrays to regenerate plots, reusing the same stub keeps regenerated artifacts easy
        to match to the original run.

    Examples:
        >>> from pathlib import Path
        >>> _infer_label_stub_from_npz_path(Path("dLVR_arrays_static_test_runs10_B2to16_pid123.npz"))
        'static_test'
    """
    name = npz_path.name
    prefix = "dLVR_arrays_"
    marker = "_runs"
    if name.startswith(prefix) and marker in name:
        stub = name[len(prefix) :].split(marker, 1)[0]
        return stub if stub else npz_path.stem
    return npz_path.stem


def _infer_pid_from_npz_path(npz_path: Path) -> int:
    """Infer the PID embedded in an NPZ filename.

    Parameters:
        npz_path: Path to a `dLVR_arrays_*.npz` file produced by this script.

    Returns:
        int: The inferred PID if present, otherwise the current process PID.

    Notes:
        This is used only for naming regenerated plot/CSV artifacts. Falling back to the
        current PID avoids hard failures when NPZ files are renamed manually.

    Examples:
        >>> from pathlib import Path
        >>> _infer_pid_from_npz_path(Path("dLVR_arrays_static_test_runs10_B2to16_pid123.npz"))
        123
    """
    stem = npz_path.stem
    marker = "_pid"
    if marker in stem:
        tail = stem.rsplit(marker, 1)[1]
        if tail.isdigit():
            return int(tail)
    return int(os.getpid())


def main() -> None:
    """Run the block size sweep and write plots + data to disk.

    Parameters:
        None.

    Returns:
        None.

    Notes:
        - Sets `visualize=False` and `verbose=False` for the sweep runs.
        - Uses `light_mode=False` because LVR series are disabled in light_mode.
        - Deletes per-run artifacts unless `--keep-run-artifacts` is passed.

    Examples:
        >>> main()
    """
    args = parse_args()
    plot_violin = bool(getattr(args, "plot_violin_plot", False))
    plot_means = bool(getattr(args, "plot_means", False))
    plot_95_interval = bool(getattr(args, "plot_95_interval", False))
    # Default: always plot medians; other layers are optional.
    plot_medians = True

    # Backward-compatible flags.
    if bool(getattr(args, "plot_medians_mean", False)):
        plot_means = True
    if bool(getattr(args, "plot_only_medians", False)):
        plot_violin = False
        plot_medians = True
        plot_means = False

    if args.load_npz is not None:
        npz_path = args.load_npz.expanduser().resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ: {npz_path}")

        out_root = npz_path.parent
        out_root.mkdir(parents=True, exist_ok=True)

        label_stub = _infer_label_stub_from_npz_path(npz_path)
        pid = _infer_pid_from_npz_path(npz_path)

        with np.load(npz_path) as data:
            if "block_times" not in data or "seeds" not in data:
                raise ValueError("NPZ must contain keys: 'block_times' and 'seeds'.")

            block_times = [int(x) for x in np.asarray(data["block_times"], dtype=int).ravel().tolist()]
            seeds = [int(x) for x in np.asarray(data["seeds"], dtype=int).ravel().tolist()]
            runs = int(len(seeds))

            skip_step = int(np.asarray(data.get("skip_step", 0), dtype=int).reshape(()))
            if "fee_definition" in data:
                fee_definition = str(np.asarray(data["fee_definition"]).reshape(()))
            else:
                # Older NPZ files predate the fee-definition switch and always used
                # mark-to-market fee value deltas.
                fee_definition = "mtm"

            enabled_cohorts = [spec for spec in COHORT_SPECS if f"dLVR_{spec.name}" in data]
            if not enabled_cohorts:
                raise ValueError("NPZ does not contain any `dLVR_<cohort>` arrays to plot.")
            cohort_names = [spec.name for spec in enabled_cohorts]

            delta_arrays = {c: np.asarray(data[f"dLVR_{c}"], dtype=float) for c in cohort_names}

            ratio_cohort_names = [c for c in cohort_names if f"dLVR_over_dFees_{c}" in data]
            ratio_enabled_cohorts = [spec for spec in enabled_cohorts if spec.name in ratio_cohort_names]
            ratio_arrays = {c: np.asarray(data[f"dLVR_over_dFees_{c}"], dtype=float) for c in ratio_cohort_names}

            if "jit_success_mask" in data:
                jit_success_arrays: Optional[np.ndarray] = np.asarray(data["jit_success_mask"], dtype=np.uint8)
            else:
                jit_success_arrays = None

        block_idx = {int(b): i for i, b in enumerate(block_times)}
        expected_b = len(block_times)
        expected_s = len(seeds)

        for cohort, arr in delta_arrays.items():
            if arr.ndim != 3 or arr.shape[0] != expected_b or arr.shape[1] != expected_s:
                raise ValueError(
                    f"Bad shape for dLVR_{cohort}: got {arr.shape}, expected ({expected_b}, {expected_s}, TΔ)."
                )
        for cohort, arr in ratio_arrays.items():
            if arr.ndim != 3 or arr.shape[0] != expected_b or arr.shape[1] != expected_s:
                raise ValueError(
                    f"Bad shape for dLVR_over_dFees_{cohort}: got {arr.shape}, expected ({expected_b}, {expected_s}, TΔ)."
                )

        if jit_success_arrays is not None and "jiter" in cohort_names:
            ref = delta_arrays.get("jiter")
            if ref is None or jit_success_arrays.ndim != 3 or jit_success_arrays.shape[:2] != ref.shape[:2]:
                print("[warn] Ignoring jit_success_mask due to incompatible shape.")
                jit_success_arrays = None
            elif jit_success_arrays.shape[2] != ref.shape[2]:
                # Best-effort alignment when the NPZ was produced with a different expected Δ length.
                n = min(int(jit_success_arrays.shape[2]), int(ref.shape[2]))
                jit_success_arrays = jit_success_arrays[:, :, :n]
                delta_arrays["jiter"] = ref[:, :, :n]
                if "jiter" in ratio_arrays:
                    ratio_arrays["jiter"] = ratio_arrays["jiter"][:, :, :n]
        else:
            jit_success_arrays = None

        if set(ratio_cohort_names) != set(cohort_names):
            missing = sorted(set(cohort_names) - set(ratio_cohort_names))
            if missing:
                print(f"[warn] NPZ is missing ΔLVR/ΔFees arrays for cohorts: {', '.join(missing)}")

        print(f"[LVR_vs_blocksize] loaded: {npz_path}")
        print(f"[LVR_vs_blocksize] fee definition: {fee_definition}")
        print(f"[LVR_vs_blocksize] cohorts: {', '.join(cohort_names)}")
        print(f"[LVR_vs_blocksize] B grid: {block_times[0]}..{block_times[-1]} (n={len(block_times)})")
        print(f"[LVR_vs_blocksize] seeds:  {seeds[0]}..{seeds[-1]} (n={len(seeds)})")

        lvr_yaxis_title = "ΔLVR (block)" if str(fee_definition) == "flow" else "ΔLVR"
        ratio_yaxis_title = "ΔLVR/ΔFees" if str(fee_definition) == "flow" else "ΔLVR/ΔFees (mtm)"

        # Build distributions per (cohort, B) by pooling across seeds and time.
        distributions: Dict[str, Dict[int, np.ndarray]] = {c: {} for c in cohort_names}
        summary_rows: List[Dict[str, Any]] = []
        for cohort in cohort_names:
            arr = delta_arrays[cohort]
            for b in block_times:
                bi = block_idx[int(b)]
                flat = arr[bi, :, :].reshape(-1)
                if cohort == "jiter" and jit_success_arrays is not None:
                    m = jit_success_arrays[bi, :, :].reshape(-1).astype(bool)
                    flat = flat[m]
                flat = flat[np.isfinite(flat)]
                if bool(plot_violin):
                    distributions[cohort][int(b)] = flat
                if flat.size == 0:
                    stats = dict(
                        n=0,
                        mean=np.nan,
                        std=np.nan,
                        median=np.nan,
                        p2_5=np.nan,
                        p25=np.nan,
                        p75=np.nan,
                        p97_5=np.nan,
                    )
                else:
                    p2_5, p25, p75, p97_5 = [float(x) for x in np.percentile(flat, [2.5, 25.0, 75.0, 97.5])]
                    stats = dict(
                        n=int(flat.size),
                        mean=float(np.mean(flat)),
                        std=float(np.std(flat)),
                        median=float(np.median(flat)),
                        p2_5=p2_5,
                        p25=p25,
                        p75=p75,
                        p97_5=p97_5,
                    )
                summary_rows.append(
                    {
                        "cohort": cohort,
                        "block_time": int(b),
                        **stats,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_root / f"dLVR_summary_{label_stub}_runs{runs}_pid{pid}.csv"
        summary_df.to_csv(summary_csv, index=False)

        ratio_distributions: Optional[Dict[str, Dict[int, np.ndarray]]] = None
        ratio_summary_df: Optional[pd.DataFrame] = None
        ratio_summary_csv: Optional[Path] = None
        if ratio_cohort_names:
            ratio_distributions = {c: {} for c in ratio_cohort_names}
            ratio_summary_rows: List[Dict[str, Any]] = []
            for cohort in ratio_cohort_names:
                arr = ratio_arrays[cohort]
                for b in block_times:
                    bi = block_idx[int(b)]
                    flat = arr[bi, :, :].reshape(-1)
                    if cohort == "jiter" and jit_success_arrays is not None:
                        m = jit_success_arrays[bi, :, :].reshape(-1).astype(bool)
                        flat = flat[m]
                    flat = flat[np.isfinite(flat)]
                    if bool(plot_violin):
                        ratio_distributions[cohort][int(b)] = flat
                    if flat.size == 0:
                        stats = dict(
                            n=0,
                            mean=np.nan,
                            std=np.nan,
                            median=np.nan,
                            p2_5=np.nan,
                            p25=np.nan,
                            p75=np.nan,
                            p97_5=np.nan,
                        )
                    else:
                        p2_5, p25, p75, p97_5 = [
                            float(x) for x in np.percentile(flat, [2.5, 25.0, 75.0, 97.5])
                        ]
                        stats = dict(
                            n=int(flat.size),
                            mean=float(np.mean(flat)),
                            std=float(np.std(flat)),
                            median=float(np.median(flat)),
                            p2_5=p2_5,
                            p25=p25,
                            p75=p75,
                            p97_5=p97_5,
                        )
                    ratio_summary_rows.append(
                        {
                            "cohort": cohort,
                            "block_time": int(b),
                            **stats,
                        }
                    )

            ratio_summary_df = pd.DataFrame(ratio_summary_rows)
            ratio_summary_csv = out_root / f"dLVR_over_dFees_summary_{label_stub}_runs{runs}_pid{pid}.csv"
            ratio_summary_df.to_csv(ratio_summary_csv, index=False)

        plot_label = "violin" if bool(plot_violin) else "medians"
        if not bool(plot_violin):
            medians_by_cohort: Dict[str, List[float]] = {}
            means_by_cohort: Dict[str, List[float]] = {}
            for cohort in cohort_names:
                cohort_df = summary_df[summary_df["cohort"] == cohort].sort_values("block_time")
                medians_by_cohort[cohort] = [float(x) for x in cohort_df["median"].to_list()]
                means_by_cohort[cohort] = [float(x) for x in cohort_df["mean"].to_list()]
            p2_5_by_cohort: Optional[Dict[str, List[float]]] = None
            p97_5_by_cohort: Optional[Dict[str, List[float]]] = None
            if bool(plot_95_interval) and bool(plot_medians):
                p2_5_by_cohort = {}
                p97_5_by_cohort = {}
                for cohort in cohort_names:
                    cohort_df = summary_df[summary_df["cohort"] == cohort].sort_values("block_time")
                    p2_5_by_cohort[cohort] = [float(x) for x in cohort_df["p2_5"].to_list()]
                    p97_5_by_cohort[cohort] = [float(x) for x in cohort_df["p97_5"].to_list()]
            fig = _build_medians_only_figure(
                medians_by_cohort,
                means=means_by_cohort,
                p2_5=p2_5_by_cohort,
                p97_5=p97_5_by_cohort,
                cohort_specs=enabled_cohorts,
                block_times=block_times,
                yaxis_title=lvr_yaxis_title,
                plot_medians=bool(plot_medians),
                plot_95_interval=bool(plot_95_interval),
                plot_means=bool(plot_means),
            )
            html_path = out_root / f"dLVR_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.html"
            png_path = out_root / f"dLVR_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.png"

            ratio_fig: Optional[go.Figure] = None
            ratio_html_path: Optional[Path] = None
            ratio_png_path: Optional[Path] = None
            if ratio_summary_df is not None:
                ratio_medians_by_cohort: Dict[str, List[float]] = {}
                ratio_means_by_cohort: Dict[str, List[float]] = {}
                for cohort in ratio_cohort_names:
                    cohort_df = ratio_summary_df[ratio_summary_df["cohort"] == cohort].sort_values("block_time")
                    ratio_medians_by_cohort[cohort] = [float(x) for x in cohort_df["median"].to_list()]
                    ratio_means_by_cohort[cohort] = [float(x) for x in cohort_df["mean"].to_list()]
                ratio_p2_5_by_cohort: Optional[Dict[str, List[float]]] = None
                ratio_p97_5_by_cohort: Optional[Dict[str, List[float]]] = None
                if bool(plot_95_interval) and bool(plot_medians):
                    ratio_p2_5_by_cohort = {}
                    ratio_p97_5_by_cohort = {}
                    for cohort in ratio_cohort_names:
                        cohort_df = ratio_summary_df[ratio_summary_df["cohort"] == cohort].sort_values("block_time")
                        ratio_p2_5_by_cohort[cohort] = [float(x) for x in cohort_df["p2_5"].to_list()]
                        ratio_p97_5_by_cohort[cohort] = [float(x) for x in cohort_df["p97_5"].to_list()]
                ratio_fig = _build_medians_only_figure(
                    ratio_medians_by_cohort,
                    means=ratio_means_by_cohort,
                    p2_5=ratio_p2_5_by_cohort,
                    p97_5=ratio_p97_5_by_cohort,
                    cohort_specs=ratio_enabled_cohorts,
                    block_times=block_times,
                    yaxis_title=ratio_yaxis_title,
                    plot_medians=bool(plot_medians),
                    plot_95_interval=bool(plot_95_interval),
                    plot_means=bool(plot_means),
                )
                ratio_html_path = out_root / (
                    f"dLVR_over_dFees_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.html"
                )
                ratio_png_path = out_root / f"dLVR_over_dFees_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.png"
        else:
            fig = _build_violin_figure(
                distributions,
                cohort_specs=enabled_cohorts,
                block_times=block_times,
                yaxis_title=lvr_yaxis_title,
                plot_medians=bool(plot_medians),
                plot_95_interval=bool(plot_95_interval),
                plot_means=bool(plot_means),
            )
            html_path = out_root / f"dLVR_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.html"
            png_path = out_root / f"dLVR_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.png"

            ratio_fig = None
            ratio_html_path = None
            ratio_png_path = None
            if ratio_distributions is not None:
                ratio_fig = _build_violin_figure(
                    ratio_distributions,
                    cohort_specs=ratio_enabled_cohorts,
                    block_times=block_times,
                    yaxis_title=ratio_yaxis_title,
                    plot_medians=bool(plot_medians),
                    plot_95_interval=bool(plot_95_interval),
                    plot_means=bool(plot_means),
                )
                ratio_html_path = out_root / (
                    f"dLVR_over_dFees_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.html"
                )
                ratio_png_path = out_root / f"dLVR_over_dFees_{plot_label}_vs_block_time_{label_stub}_runs{runs}_pid{pid}.png"

        fig.write_html(html_path)
        if ratio_fig is not None and ratio_html_path is not None:
            ratio_fig.write_html(ratio_html_path)

        try:
            fig.write_image(png_path)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] PNG export failed (is kaleido installed?): {exc}")
        try:
            if ratio_fig is not None and ratio_png_path is not None:
                ratio_fig.write_image(ratio_png_path)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] PNG export failed (is kaleido installed?): {exc}")

        print(f"[LVR_vs_blocksize] wrote: {summary_csv}")
        print(f"[LVR_vs_blocksize] wrote: {html_path}")
        if png_path.exists():
            print(f"[LVR_vs_blocksize] wrote: {png_path}")
        if ratio_summary_csv is not None:
            print(f"[LVR_vs_blocksize] wrote: {ratio_summary_csv}")
        if ratio_html_path is not None:
            print(f"[LVR_vs_blocksize] wrote: {ratio_html_path}")
        if ratio_png_path is not None and ratio_png_path.exists():
            print(f"[LVR_vs_blocksize] wrote: {ratio_png_path}")
        return

    if args.runs <= 0:
        raise SystemExit("--runs must be positive.")
    if args.seed_step <= 0:
        raise SystemExit("--seed-step must be positive.")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive.")
    if args.block_min < 2:
        raise SystemExit("--block-min must be >= 2 (run.py requires block_time > 1).")
    if args.block_min > args.block_max:
        raise SystemExit("--block-min cannot exceed --block-max.")

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    _, base_params = load_simulation_parameters(config_path, simulate_func=simulate)
    base_params = dict(base_params)

    # We need LVR recorders => light_mode must be False.
    base_params["light_mode"] = False
    # Avoid per-run plots; we only want the aggregated violin plot.
    base_params["visualize"] = False
    # Reduce stdout; note: detailed per-run logs are controlled by light_mode.
    base_params["verbose"] = False

    enabled_cohorts = _resolve_enabled_cohorts(base_params)
    if not enabled_cohorts:
        raise SystemExit("No enabled cohorts detected (active/passive/jiter are all disabled).")
    cohort_names = [c.name for c in enabled_cohorts]

    scenario_root = scenario_output_root(config_path)
    out_root = scenario_root / "lvr_vs_blocksize"
    out_root.mkdir(parents=True, exist_ok=True)
    fee_mode_label = str(base_params.get("fee_mode", "unknown"))
    fee_def_label = str(args.fee_definition)
    lvr_yaxis_title = "ΔLVR (block)" if fee_def_label == "flow" else "ΔLVR"
    ratio_yaxis_title = "ΔLVR/ΔFees" if fee_def_label == "flow" else "ΔLVR/ΔFees (mtm)"
    pid = int(os.getpid())

    # Temp root for per-run artifacts (logs, etc.). Each worker uses a unique folder.
    tmp_base = out_root / "_tmp_runs"
    run_tmp_root = _make_unique_dir(tmp_base / str(os.getpid()))

    block_times = list(range(int(args.block_min), int(args.block_max) + 1))
    seed0 = int(args.seed_base) if args.seed_base is not None else int(base_params.get("seed", 1))
    seeds = [int(seed0 + i * int(args.seed_step)) for i in range(int(args.runs))]

    # Expected Δ series length (used only for saving aligned 3D arrays).
    T = int(base_params.get("T", 0))
    skip_step = int(base_params.get("skip_step", 0))
    expected_delta_len = max(0, (T - skip_step) - 1)

    # Pre-allocate (B, seed, step) arrays per cohort for reproducible storage.
    block_idx = {int(b): i for i, b in enumerate(block_times)}
    seed_idx = {int(s): i for i, s in enumerate(seeds)}
    delta_arrays: Dict[str, np.ndarray] = {
        c.name: np.full((len(block_times), len(seeds), expected_delta_len), np.nan, dtype=float)
        for c in enabled_cohorts
    }
    ratio_arrays: Dict[str, np.ndarray] = {
        c.name: np.full((len(block_times), len(seeds), expected_delta_len), np.nan, dtype=float)
        for c in enabled_cohorts
    }
    jit_success_arrays: Optional[np.ndarray]
    if "jiter" in cohort_names:
        jit_success_arrays = np.zeros((len(block_times), len(seeds), expected_delta_len), dtype=np.uint8)
    else:
        jit_success_arrays = None

    # Run all (B, seed) combinations in parallel.
    tasks: List[Tuple[int, int]] = [(b, s) for b in block_times for s in seeds]
    print(f"[LVR_vs_blocksize] config: {config_path}")
    print(f"[LVR_vs_blocksize] fee mode: {fee_mode_label}")
    print(f"[LVR_vs_blocksize] fee definition: {fee_def_label}")
    print(f"[LVR_vs_blocksize] cohorts: {', '.join(cohort_names)}")
    print(f"[LVR_vs_blocksize] B grid: {block_times[0]}..{block_times[-1]} (n={len(block_times)})")
    print(f"[LVR_vs_blocksize] seeds:  {seeds[0]}..{seeds[-1]} (n={len(seeds)}, step={args.seed_step})")
    print(f"[LVR_vs_blocksize] total runs: {len(tasks)}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = [
            executor.submit(
                _run_one,
                seed,
                block_time,
                base_params=base_params,
                tmp_root=run_tmp_root,
                keep_run_artifacts=bool(args.keep_run_artifacts),
                cohort_names=cohort_names,
                fee_definition=fee_def_label,
            )
            for (block_time, seed) in tasks
        ]
        with tqdm(total=len(futures), desc="Runs", unit="sim") as pbar:
            for fut in as_completed(futures):
                results.append(fut.result())
                pbar.update(1)

    # Fill aligned arrays.
    for r in results:
        b = int(r["block_time"])
        s = int(r["seed"])
        bi = block_idx.get(b)
        si = seed_idx.get(s)
        if bi is None or si is None:
            continue
        for cohort in cohort_names:
            key = f"dLVR_{cohort}"
            if key not in r:
                continue
            vals = np.asarray(r[key], dtype=float)
            if vals.size == 0 or expected_delta_len <= 0:
                continue
            n = min(int(vals.size), int(expected_delta_len))
            delta_arrays[cohort][bi, si, :n] = vals[:n]
        for cohort in cohort_names:
            key = f"dLVR_over_dFees_{cohort}"
            if key not in r:
                continue
            vals = np.asarray(r[key], dtype=float)
            if vals.size == 0 or expected_delta_len <= 0:
                continue
            n = min(int(vals.size), int(expected_delta_len))
            ratio_arrays[cohort][bi, si, :n] = vals[:n]
        if jit_success_arrays is not None and "jit_success_mask" in r:
            mask = np.asarray(r["jit_success_mask"], dtype=bool)
            if mask.size > 0 and expected_delta_len > 0:
                n = min(int(mask.size), int(expected_delta_len))
                jit_success_arrays[bi, si, :n] = mask[:n].astype(np.uint8)

    # Save arrays for reproducibility (time series per run).
    npz_path = out_root / (
        f"dLVR_arrays_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}"
        f"_B{block_times[0]}to{block_times[-1]}_pid{pid}.npz"
    )
    npz_payload: Dict[str, Any] = {
        "block_times": np.array(block_times, dtype=int),
        "seeds": np.array(seeds, dtype=int),
        "T": np.array(T, dtype=int),
        "skip_step": np.array(skip_step, dtype=int),
        "fee_definition": np.array(fee_def_label),
    }
    for cohort in cohort_names:
        npz_payload[f"dLVR_{cohort}"] = delta_arrays[cohort]
        npz_payload[f"dLVR_over_dFees_{cohort}"] = ratio_arrays[cohort]
    if jit_success_arrays is not None:
        npz_payload["jit_success_mask"] = jit_success_arrays
    np.savez_compressed(npz_path, **npz_payload)

    # Build distributions per (cohort, B) by pooling across seeds and time.
    distributions: Dict[str, Dict[int, np.ndarray]] = {c: {} for c in cohort_names}
    summary_rows: List[Dict[str, Any]] = []
    for cohort in cohort_names:
        arr = delta_arrays[cohort]
        for b in block_times:
            bi = block_idx[int(b)]
            flat = arr[bi, :, :].reshape(-1)
            if cohort == "jiter" and jit_success_arrays is not None:
                m = jit_success_arrays[bi, :, :].reshape(-1).astype(bool)
                flat = flat[m]
            flat = flat[np.isfinite(flat)]
            if bool(plot_violin):
                distributions[cohort][int(b)] = flat
            if flat.size == 0:
                stats = dict(
                    n=0,
                    mean=np.nan,
                    std=np.nan,
                    median=np.nan,
                    p2_5=np.nan,
                    p25=np.nan,
                    p75=np.nan,
                    p97_5=np.nan,
                )
            else:
                p2_5, p25, p75, p97_5 = [float(x) for x in np.percentile(flat, [2.5, 25.0, 75.0, 97.5])]
                stats = dict(
                    n=int(flat.size),
                    mean=float(np.mean(flat)),
                    std=float(np.std(flat)),
                    median=float(np.median(flat)),
                    p2_5=p2_5,
                    p25=p25,
                    p75=p75,
                    p97_5=p97_5,
                )
            summary_rows.append(
                {
                    "cohort": cohort,
                    "block_time": int(b),
                    **stats,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / (
        f"dLVR_summary_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.csv"
    )
    summary_df.to_csv(summary_csv, index=False)

    # Build distributions/stats for ΔLVR/ΔFees ("fee coverage") ratio.
    ratio_distributions: Dict[str, Dict[int, np.ndarray]] = {c: {} for c in cohort_names}
    ratio_summary_rows: List[Dict[str, Any]] = []
    for cohort in cohort_names:
        arr = ratio_arrays[cohort]
        for b in block_times:
            bi = block_idx[int(b)]
            flat = arr[bi, :, :].reshape(-1)
            if cohort == "jiter" and jit_success_arrays is not None:
                m = jit_success_arrays[bi, :, :].reshape(-1).astype(bool)
                flat = flat[m]
            flat = flat[np.isfinite(flat)]
            if bool(plot_violin):
                ratio_distributions[cohort][int(b)] = flat
            if flat.size == 0:
                stats = dict(
                    n=0,
                    mean=np.nan,
                    std=np.nan,
                    median=np.nan,
                    p2_5=np.nan,
                    p25=np.nan,
                    p75=np.nan,
                    p97_5=np.nan,
                )
            else:
                p2_5, p25, p75, p97_5 = [float(x) for x in np.percentile(flat, [2.5, 25.0, 75.0, 97.5])]
                stats = dict(
                    n=int(flat.size),
                    mean=float(np.mean(flat)),
                    std=float(np.std(flat)),
                    median=float(np.median(flat)),
                    p2_5=p2_5,
                    p25=p25,
                    p75=p75,
                    p97_5=p97_5,
                )
            ratio_summary_rows.append(
                {
                    "cohort": cohort,
                    "block_time": int(b),
                    **stats,
                }
            )

    ratio_summary_df = pd.DataFrame(ratio_summary_rows)
    ratio_summary_csv = out_root / (
        f"dLVR_over_dFees_summary_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.csv"
    )
    ratio_summary_df.to_csv(ratio_summary_csv, index=False)

    plot_label = "violin" if bool(plot_violin) else "medians"
    # fig_title = f"ΔLVR per block vs block_time (runs={int(args.runs)}, skip_step={skip_step})"
    if not bool(plot_violin):
        medians_by_cohort: Dict[str, List[float]] = {}
        means_by_cohort: Dict[str, List[float]] = {}
        for cohort in cohort_names:
            cohort_df = summary_df[summary_df["cohort"] == cohort].sort_values("block_time")
            medians_by_cohort[cohort] = [float(x) for x in cohort_df["median"].to_list()]
            means_by_cohort[cohort] = [float(x) for x in cohort_df["mean"].to_list()]
        p2_5_by_cohort = None
        p97_5_by_cohort = None
        if bool(plot_95_interval) and bool(plot_medians):
            p2_5_by_cohort = {}
            p97_5_by_cohort = {}
            for cohort in cohort_names:
                cohort_df = summary_df[summary_df["cohort"] == cohort].sort_values("block_time")
                p2_5_by_cohort[cohort] = [float(x) for x in cohort_df["p2_5"].to_list()]
                p97_5_by_cohort[cohort] = [float(x) for x in cohort_df["p97_5"].to_list()]
        fig = _build_medians_only_figure(
            medians_by_cohort,
            means=means_by_cohort,
            p2_5=p2_5_by_cohort,
            p97_5=p97_5_by_cohort,
            cohort_specs=enabled_cohorts,
            block_times=block_times,
            yaxis_title=lvr_yaxis_title,
            plot_medians=bool(plot_medians),
            plot_95_interval=bool(plot_95_interval),
            plot_means=bool(plot_means),
            # title=fig_title,
        )
        html_path = out_root / (
            f"dLVR_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.html"
        )
        png_path = out_root / (
            f"dLVR_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.png"
        )

        ratio_medians_by_cohort: Dict[str, List[float]] = {}
        ratio_means_by_cohort: Dict[str, List[float]] = {}
        for cohort in cohort_names:
            cohort_df = ratio_summary_df[ratio_summary_df["cohort"] == cohort].sort_values("block_time")
            ratio_medians_by_cohort[cohort] = [float(x) for x in cohort_df["median"].to_list()]
            ratio_means_by_cohort[cohort] = [float(x) for x in cohort_df["mean"].to_list()]
        ratio_p2_5_by_cohort = None
        ratio_p97_5_by_cohort = None
        if bool(plot_95_interval) and bool(plot_medians):
            ratio_p2_5_by_cohort = {}
            ratio_p97_5_by_cohort = {}
            for cohort in cohort_names:
                cohort_df = ratio_summary_df[ratio_summary_df["cohort"] == cohort].sort_values("block_time")
                ratio_p2_5_by_cohort[cohort] = [float(x) for x in cohort_df["p2_5"].to_list()]
                ratio_p97_5_by_cohort[cohort] = [float(x) for x in cohort_df["p97_5"].to_list()]
        ratio_fig = _build_medians_only_figure(
            ratio_medians_by_cohort,
            means=ratio_means_by_cohort,
            p2_5=ratio_p2_5_by_cohort,
            p97_5=ratio_p97_5_by_cohort,
            cohort_specs=enabled_cohorts,
            block_times=block_times,
            yaxis_title=ratio_yaxis_title,
            plot_medians=bool(plot_medians),
            plot_95_interval=bool(plot_95_interval),
            plot_means=bool(plot_means),
        )
        ratio_html_path = out_root / (
            f"dLVR_over_dFees_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.html"
        )
        ratio_png_path = out_root / (
            f"dLVR_over_dFees_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.png"
        )
    else:
        fig = _build_violin_figure(
            distributions,
            cohort_specs=enabled_cohorts,
            block_times=block_times,
            yaxis_title=lvr_yaxis_title,
            plot_medians=bool(plot_medians),
            plot_95_interval=bool(plot_95_interval),
            plot_means=bool(plot_means),
            # title=fig_title,
        )
        html_path = out_root / (
            f"dLVR_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.html"
        )
        png_path = out_root / (
            f"dLVR_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.png"
        )

        ratio_fig = _build_violin_figure(
            ratio_distributions,
            cohort_specs=enabled_cohorts,
            block_times=block_times,
            yaxis_title=ratio_yaxis_title,
            plot_medians=bool(plot_medians),
            plot_95_interval=bool(plot_95_interval),
            plot_means=bool(plot_means),
        )
        ratio_html_path = out_root / (
            f"dLVR_over_dFees_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.html"
        )
        ratio_png_path = out_root / (
            f"dLVR_over_dFees_{plot_label}_vs_block_time_{fee_mode_label}_{fee_def_label}_{config_path.stem}_runs{int(args.runs)}_pid{pid}.png"
        )

    fig.write_html(html_path)
    ratio_fig.write_html(ratio_html_path)

    try:
        fig.write_image(png_path)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] PNG export failed (is kaleido installed?): {exc}")
    try:
        ratio_fig.write_image(ratio_png_path)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] PNG export failed (is kaleido installed?): {exc}")

    if not args.keep_run_artifacts:
        shutil.rmtree(run_tmp_root, ignore_errors=True)
        try:
            tmp_base.rmdir()
        except OSError:
            pass

    print(f"[LVR_vs_blocksize] wrote: {npz_path}")
    print(f"[LVR_vs_blocksize] wrote: {summary_csv}")
    print(f"[LVR_vs_blocksize] wrote: {html_path}")
    if png_path.exists():
        print(f"[LVR_vs_blocksize] wrote: {png_path}")
    print(f"[LVR_vs_blocksize] wrote: {ratio_summary_csv}")
    print(f"[LVR_vs_blocksize] wrote: {ratio_html_path}")
    if ratio_png_path.exists():
        print(f"[LVR_vs_blocksize] wrote: {ratio_png_path}")


if __name__ == "__main__":
    main()
