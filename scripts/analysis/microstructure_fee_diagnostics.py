"""Generate Section 4.1 static-vs-dynamic fee microstructure diagnostics.

This script reruns the two paper-stable Model 0 scenarios used in Section 4.1
with identical seeds and parameters except for the fee controller, then writes:

- a side-by-side CEX/DEX price zoom with the no-arbitrage fee band;
- a side-by-side DEX end-of-block return ACF comparison;
- machine-readable CSVs containing the plotted ACF values, zoom data, and
  summary counts.

The scientific contrast is model-conditional: toxicity fees widen the effective
no-arbitrage band in high-toxicity periods, so fewer arbitrage intents are
executed and the static-fee lag-1 mean-reversion signature is attenuated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.artifacts import build_run_manifest, safe_tag, write_json  # noqa: E402
from core.utils import load_simulation_parameters  # noqa: E402
from scripts.run import simulate  # noqa: E402


@dataclass(frozen=True)
class DiagnosticRun:
    """Container for one fee-mode run used by the paper diagnostic plots."""

    label: str
    config_path: Path
    params: Mapping[str, Any]
    output: Mapping[str, Any]
    run_root: Path


DEFAULT_CONFIGS = {
    "static": _REPO_ROOT / "abm_results/scenarios/section4_microstructure_model0_static.yml",
    "toxicity": _REPO_ROOT / "abm_results/scenarios/section4_microstructure_model0_toxicity.yml",
}
DEFAULT_IMAGE_DIR = _REPO_ROOT / "paper/images"
DEFAULT_TABLE_DIR = _REPO_ROOT / "paper/tables"
DEFAULT_ARTIFACT_ROOT = _REPO_ROOT / "abm_results/scenarios/section4_microstructure_diagnostics/runs"


COLOR_CEX = "#8b5cf6"
COLOR_DEX = "#00b894"
COLOR_ACF = "#636EFA"
COLOR_BAND = "#d9d9d9"


def _finite_positive(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def log_return_acf(prices: Sequence[float], *, skip: int, max_lag: int) -> np.ndarray:
    """Compute sample ACF of log returns, omitting lag 0.

    Parameters
    ----------
    prices:
        Positive price series. Non-positive or non-finite values are dropped.
    skip:
        Number of initial blocks removed before computing returns.
    max_lag:
        Maximum lag in blocks for the returned ACF values.
    """

    returns = _clean_log_returns(prices, skip=skip)
    if returns.size < 3:
        return np.array([], dtype=float)

    lag_cap = max(1, min(int(max_lag), int(returns.size - 1)))
    centered = returns - float(np.mean(returns))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.array([], dtype=float)

    out = np.empty(lag_cap, dtype=float)
    for lag in range(1, lag_cap + 1):
        out[lag - 1] = float(np.dot(centered[:-lag], centered[lag:]) / denom)
    return out


def _clean_log_returns(prices: Sequence[float], *, skip: int) -> np.ndarray:
    """Return finite log returns after the configured transient skip."""

    price_arr = np.asarray(prices, dtype=float)
    s0 = max(0, min(int(skip), int(price_arr.size)))
    clean = _finite_positive(price_arr[s0:])
    returns = np.diff(np.log(clean))
    return returns[np.isfinite(returns)]


def _acf_no_correlation_band(sample_size: int, confidence_level: float = 0.95) -> float:
    """Analytic two-sided no-correlation ACF band half-width.

    Under the white-noise null, sample autocorrelations at fixed non-zero lags
    are approximately N(0, 1/N). The plotted band is therefore
    z_{1-alpha/2}/sqrt(N), without bootstrap or permutation resampling.
    """

    n = int(sample_size)
    if n <= 0:
        raise ValueError(f"ACF no-correlation band requires positive sample size, got {sample_size}")
    from statistics import NormalDist

    z_value = NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)
    return float(z_value / np.sqrt(n))


def _as_float_array(output: Mapping[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(output[key], dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array for {key}, got shape {arr.shape}")
    return arr


def _run_config(label: str, config_path: Path, *, artifact_root: Path, force: bool) -> DiagnosticRun:
    """Run a configured scenario and return full in-memory diagnostics.

    A compact JSON metadata file is written under ``artifact_root`` so the run
    has inspectable provenance even though the raw runs directory is ignored by
    git.
    """

    cfg = config_path.expanduser().resolve()
    scenario_label, params = load_simulation_parameters(cfg, simulate_func=simulate)
    params = dict(params)
    fee_mode = str(params.get("fee_mode", scenario_label))
    if fee_mode != label:
        raise ValueError(f"Config {cfg} has fee_mode={fee_mode!r}, expected {label!r}")

    run_id = safe_tag(f"{label}_seed{int(params['seed'])}_T{int(params['T'])}")
    run_root = artifact_root / run_id
    if run_root.exists() and any(run_root.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic run directory: {run_root}. "
            "Use --force to regenerate."
        )
    run_root.mkdir(parents=True, exist_ok=True)

    # The paper figures are generated by this script, not by the generic Plotly
    # output stack. Keep simulate() non-visual and quiet, but still write its
    # lightweight output_data arrays under the run root for inspection.
    params["visualize"] = False
    params["verbose"] = False
    params["light_mode"] = False
    params["results_root"] = run_root

    output = simulate(**params)

    manifest = build_run_manifest(
        script="scripts.analysis.microstructure_fee_diagnostics",
        run_id=run_id,
        config_path=cfg,
    )
    write_json(
        run_root / "metadata.json",
        {
            **manifest.to_dict(),
            "scenario_label": scenario_label,
            "fee_mode": fee_mode,
            "seed": int(params["seed"]),
            "T": int(params["T"]),
            "skip_step": int(params["skip_step"]),
            "config_path": str(cfg),
            "run_root": str(run_root),
        },
    )
    return DiagnosticRun(label=label, config_path=cfg, params=params, output=output, run_root=run_root)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _summary_rows(runs: Sequence[DiagnosticRun], *, max_lag: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        out = run.output
        params = run.params
        fee = np.asarray(out["fee_series"], dtype=float)
        skip = int(params["skip_step"])
        acf = log_return_acf(out["DEX_price"], skip=skip, max_lag=max_lag)
        total_arb = int(out.get("total_arb_swaps", 0))
        no_op = int(out.get("arb_no_op_in_band", 0))
        rejected = int(out.get("arb_swaps_rejected_profitability", 0))
        attempts = total_arb + no_op + rejected
        rows.append(
            {
                "fee_mode": run.label,
                "config_path": str(run.config_path.relative_to(_REPO_ROOT)),
                "run_root": str(run.run_root.relative_to(_REPO_ROOT)),
                "seed": int(params["seed"]),
                "T": int(params["T"]),
                "skip_step": skip,
                "block_time": int(params["block_time"]),
                "mean_fee_after_skip": float(np.mean(fee[skip:])),
                "median_fee_after_skip": float(np.median(fee[skip:])),
                "total_arb_swaps": total_arb,
                "arb_no_op_in_band": no_op,
                "arb_swaps_rejected_profitability": rejected,
                "arb_attempts": attempts,
                "arb_execution_share": float(total_arb / attempts) if attempts else float("nan"),
                "arb_no_op_share": float(no_op / attempts) if attempts else float("nan"),
                "dex_return_acf_lag1": float(acf[0]) if acf.size else float("nan"),
            }
        )
    return rows


def _zoom_rows(runs: Sequence[DiagnosticRun], *, zoom_start: int, zoom_end: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        dex = _as_float_array(run.output, "DEX_price")
        cex = _as_float_array(run.output, "CEX_price")
        lo = _as_float_array(run.output, "band_lo")
        hi = _as_float_array(run.output, "band_hi")
        fee = np.asarray(run.output["fee_series"], dtype=float)
        n = min(dex.size, cex.size, lo.size, hi.size, fee.size)
        start = max(0, int(zoom_start))
        end = min(n - 1, int(zoom_end))
        if end <= start:
            raise ValueError(f"Invalid zoom range after clipping: {start=} {end=} n={n}")
        for block in range(start, end + 1):
            rows.append(
                {
                    "fee_mode": run.label,
                    "block": block,
                    "cex_price": float(cex[block]),
                    "dex_price": float(dex[block]),
                    "band_low": float(lo[block]),
                    "band_high": float(hi[block]),
                    "fee": float(fee[block]),
                }
            )
    return rows


def _acf_rows(runs: Sequence[DiagnosticRun], *, max_lag: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        acf = log_return_acf(run.output["DEX_price"], skip=int(run.params["skip_step"]), max_lag=max_lag)
        for lag, value in enumerate(acf, start=1):
            rows.append({"fee_mode": run.label, "lag": lag, "acf": float(value)})
    return rows


def plot_price_zoom(
    runs: Sequence[DiagnosticRun],
    *,
    zoom_start: int,
    zoom_end: int,
    output_stem: Path,
) -> None:
    """Write the side-by-side CEX/DEX price zoom figure."""

    fig, axes = plt.subplots(1, len(runs), figsize=(12.8, 4.4), sharex=True, sharey=False)
    if len(runs) == 1:
        axes = [axes]

    titles = {
        "static": "Static fees (narrow band)",
        "toxicity": "Toxicity fees (wider band)",
    }
    for ax, run in zip(axes, runs):
        dex = _as_float_array(run.output, "DEX_price")
        cex = _as_float_array(run.output, "CEX_price")
        lo = _as_float_array(run.output, "band_lo")
        hi = _as_float_array(run.output, "band_hi")
        n = min(dex.size, cex.size, lo.size, hi.size)
        steps = np.arange(n)
        mask = (steps >= int(zoom_start)) & (steps <= int(zoom_end))
        ax.fill_between(steps[mask], lo[:n][mask], hi[:n][mask], color=COLOR_BAND, alpha=0.85, label="No-arb fee band")
        ax.plot(steps[mask], cex[:n][mask], color=COLOR_CEX, linestyle="--", linewidth=1.7, label="CEX price $m_t$")
        ax.plot(steps[mask], dex[:n][mask], color=COLOR_DEX, linewidth=1.9, label="DEX price $P_t$")
        ax.set_title(titles.get(run.label, run.label), fontsize=13)
        ax.set_xlabel("Block", fontsize=11)
        ax.grid(True, color="#bbbbbb", alpha=0.35, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel("Price", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_acf_comparison(runs: Sequence[DiagnosticRun], *, max_lag: int, output_stem: Path) -> None:
    """Write the static-vs-toxicity DEX return ACF comparison figure."""

    acfs = [log_return_acf(run.output["DEX_price"], skip=int(run.params["skip_step"]), max_lag=max_lag) for run in runs]
    band_by_label = {
        run.label: _acf_no_correlation_band(
            _clean_log_returns(run.output["DEX_price"], skip=int(run.params["skip_step"])).size
        )
        for run in runs
    }
    y_min = min(float(np.nanmin(acf)) for acf in acfs if acf.size)
    y_max = max(float(np.nanmax(acf)) for acf in acfs if acf.size)
    max_band = max(band_by_label.values()) if band_by_label else 0.0
    y_min = min(y_min, -max_band)
    y_max = max(y_max, max_band)
    pad = 0.04
    y_limits = (min(-0.02, y_min - pad), max(0.04, y_max + pad))

    fig, axes = plt.subplots(1, len(runs), figsize=(11.2, 4.0), sharex=True, sharey=True)
    if len(runs) == 1:
        axes = [axes]

    titles = {
        "static": "Static fee",
        "toxicity": "Toxicity fee",
    }
    for ax, run, acf in zip(axes, runs, acfs):
        lags = np.arange(1, acf.size + 1)
        ax.bar(lags, acf, color=COLOR_ACF, width=0.78)
        band = band_by_label.get(run.label)
        if band is not None:
            ax.axhspan(
                -band,
                band,
                color="#999999",
                alpha=0.16,
                label="95% no-correlation band",
                zorder=0,
            )
            ax.axhline(band, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
            ax.axhline(-band, color="#777777", linestyle=":", linewidth=1.0, zorder=1)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        lag1 = float(acf[0]) if acf.size else float("nan")
        ax.set_title(f"{titles.get(run.label, run.label)} (lag 1 = {lag1:.3f})", fontsize=13)
        ax.set_xlabel("Lag (blocks)", fontsize=11)
        ax.set_xlim(0.4, max_lag + 0.6)
        ax.set_ylim(*y_limits)
        ax.grid(True, axis="y", color="#bbbbbb", alpha=0.35, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel("Autocorrelation", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-config", type=Path, default=DEFAULT_CONFIGS["static"])
    parser.add_argument("--toxicity-config", type=Path, default=DEFAULT_CONFIGS["toxicity"])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--zoom-start", type=int, default=5200)
    parser.add_argument("--zoom-end", type=int, default=5260)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the ignored scenario-local diagnostic run folders.",
    )
    args = parser.parse_args()

    runs = [
        _run_config("static", args.static_config, artifact_root=args.artifact_root, force=args.force),
        _run_config("toxicity", args.toxicity_config, artifact_root=args.artifact_root, force=args.force),
    ]

    price_stem = args.image_dir / "microstructure_price_static_vs_toxicity"
    acf_stem = args.image_dir / "microstructure_acf_static_vs_toxicity"
    plot_price_zoom(runs, zoom_start=args.zoom_start, zoom_end=args.zoom_end, output_stem=price_stem)
    plot_acf_comparison(runs, max_lag=args.max_lag, output_stem=acf_stem)

    def _manifest_path(path: Path) -> str:
        """Use repo-relative paths when possible, absolute paths for temp smoke outputs."""
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(_REPO_ROOT))
        except ValueError:
            return str(resolved)

    _write_csv(
        args.table_dir / "microstructure_fee_diagnostics_values.csv",
        _summary_rows(runs, max_lag=args.max_lag),
        fieldnames=[
            "fee_mode",
            "config_path",
            "run_root",
            "seed",
            "T",
            "skip_step",
            "block_time",
            "mean_fee_after_skip",
            "median_fee_after_skip",
            "total_arb_swaps",
            "arb_no_op_in_band",
            "arb_swaps_rejected_profitability",
            "arb_attempts",
            "arb_execution_share",
            "arb_no_op_share",
            "dex_return_acf_lag1",
        ],
    )
    _write_csv(
        args.table_dir / "microstructure_price_zoom_values.csv",
        _zoom_rows(runs, zoom_start=args.zoom_start, zoom_end=args.zoom_end),
        fieldnames=["fee_mode", "block", "cex_price", "dex_price", "band_low", "band_high", "fee"],
    )
    _write_csv(
        args.table_dir / "microstructure_acf_values.csv",
        _acf_rows(runs, max_lag=args.max_lag),
        fieldnames=["fee_mode", "lag", "acf"],
    )
    write_json(
        args.table_dir / "microstructure_fee_diagnostics_manifest.json",
        {
            "script": "scripts/analysis/microstructure_fee_diagnostics.py",
            "static_config": str(Path(args.static_config).resolve().relative_to(_REPO_ROOT)),
            "toxicity_config": str(Path(args.toxicity_config).resolve().relative_to(_REPO_ROOT)),
            "price_figure_png": _manifest_path(price_stem.with_suffix(".png")),
            "price_figure_pdf": _manifest_path(price_stem.with_suffix(".pdf")),
            "acf_figure_png": _manifest_path(acf_stem.with_suffix(".png")),
            "acf_figure_pdf": _manifest_path(acf_stem.with_suffix(".pdf")),
            "zoom_start": int(args.zoom_start),
            "zoom_end": int(args.zoom_end),
            "max_lag": int(args.max_lag),
            "runs": [
                {
                    "fee_mode": run.label,
                    "config_path": str(run.config_path.relative_to(_REPO_ROOT)),
                    "run_root": str(run.run_root.relative_to(_REPO_ROOT)),
                    "seed": int(run.params["seed"]),
                }
                for run in runs
            ],
        },
    )

    print(f"Wrote {price_stem.with_suffix('.png')}")
    print(f"Wrote {acf_stem.with_suffix('.png')}")
    print(f"Wrote {args.table_dir / 'microstructure_fee_diagnostics_values.csv'}")


if __name__ == "__main__":
    main()
