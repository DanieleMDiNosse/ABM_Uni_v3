#!/usr/bin/env python3
"""Build the combined Model 2 block-size fee-coverage ratio figure.

The figure has one panel per cohort and one line per fee schedule.  It reads
paper font sizes and export dimensions from ``paper/figure_style.yml`` so the
manuscript asset is controlled by the shared paper style config rather than by
LaTeX resizing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis.paper_figure_style import load_figure_style, merge_style  # noqa: E402

DEFAULT_SOURCE_ROOT = Path("/mnt/external/scenarios")
DEFAULT_DATA_PATH = _REPO_ROOT / "paper" / "tables" / "model2_blocksize_ratio_values.csv"
DEFAULT_STYLE_PATH = _REPO_ROOT / "paper" / "figure_style.yml"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "paper" / "images" / "model2_blocksize_ratio_combined.png"

COHORTS = {
    "active": "Active LPs",
    "passive": "Passive LPs",
    "jiter": "Jiter",
}

SCHEDULES = {
    "static": {
        "label": "Static",
        "color": "#000000",
        "dash": "solid",
        "source_csv": Path(
            "model2_static/lvr_vs_blocksize/"
            "static_flow_model2_static_runs50_B2to16_seed10_step1_6/"
            "dLVR_over_dFees_summary.csv"
        ),
    },
    "toxicity": {
        "label": "Toxicity",
        "color": "#D55E00",
        "dash": "dash",
        "source_csv": Path(
            "model2_tox/lvr_vs_blocksize/"
            "toxicity_flow_model2_tox_runs50_B2to16_seed10_step1_2/"
            "dLVR_over_dFees_summary.csv"
        ),
    },
    "volatility_cex": {
        "label": "Volatility (CEX)",
        "color": "#0072B2",
        "dash": "dot",
        "source_csv": Path(
            "model2_vol_cex/lvr_vs_blocksize/"
            "volatility_cex_flow_model2_vol_cex_runs50_B2to16_seed10_step1_2/"
            "dLVR_over_dFees_summary.csv"
        ),
    },
    "volatility_dex": {
        "label": "Volatility (DEX)",
        "color": "#009E73",
        "dash": "dashdot",
        "source_csv": Path(
            "model2_vol_dex/lvr_vs_blocksize/"
            "volatility_dex_flow_model2_vol_dex_runs50_B2to16_seed10_step1_2/"
            "dLVR_over_dFees_summary.csv"
        ),
    },
    "linear_asymmetric": {
        "label": "Linear asymmetric",
        "color": "#CC79A7",
        "dash": "longdash",
        "source_csv": Path(
            "model2_linear_asym/lvr_vs_blocksize/"
            "linear_asymmetric_flow_model2_linear_asym_runs50_B2to16_seed10_step1/"
            "dLVR_over_dFees_summary.csv"
        ),
    },
}

SUMMARY_COLUMNS = ["n", "mean", "std", "median", "p2_5", "p25", "p75", "p97_5"]

DEFAULT_STYLE: Dict[str, Any] = {
    "template": "plotly_white",
    "width": 1800,
    "height": 650,
    "scale": 1,
    "font": {
        "family": "Arial, sans-serif",
        "base_size": 22,
        "axis_title_size": 24,
        "tick_size": 20,
        "legend_size": 20,
        "subplot_title_size": 24,
    },
    "line": {
        "width": 4,
        "marker_size": 9,
    },
    "margins": {
        "l": 95,
        "r": 35,
        "t": 105,
        "b": 95,
    },
}


def load_paper_style(path: Path) -> Dict[str, Any]:
    """Load the shared paper style plus this figure's overrides."""

    return load_figure_style(path, DEFAULT_STYLE, figure_key="model2_blocksize_ratio_combined")


def _schedule_config(schedule: str, style: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fee-schedule plotting config with style overrides applied."""

    configured = style.get("fee_schedules", {})
    if not isinstance(configured, Mapping):
        raise TypeError("Expected mapping for fee_schedules in paper figure style")
    override = configured.get(schedule, {})
    if not isinstance(override, Mapping):
        raise TypeError(f"Expected mapping for fee_schedules.{schedule} in paper figure style")
    return merge_style(dict(SCHEDULES[schedule]), override)


def load_ratio_data(path: Path) -> pd.DataFrame:
    """Load and validate the long-form ratio table used by the figure."""
    if not path.exists():
        raise FileNotFoundError(f"Missing ratio table: {path}")
    df = pd.read_csv(path)
    required = {"block_time", "fee_schedule", "cohort", "median_R"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Ratio table {path} is missing columns: {missing}")
    df = df.copy()
    df["block_time"] = df["block_time"].astype(int)
    df["median_R"] = df["median_R"].astype(float)
    unknown_schedules = sorted(set(df["fee_schedule"]) - set(SCHEDULES))
    if unknown_schedules:
        raise ValueError(f"Unknown fee_schedule values: {unknown_schedules}")
    unknown_cohorts = sorted(set(df["cohort"]) - set(COHORTS))
    if unknown_cohorts:
        raise ValueError(f"Unknown cohort values: {unknown_cohorts}")
    return df


def _validate_summary(df: pd.DataFrame, *, source_csv: Path, schedule: str) -> None:
    """Validate one exact dLVR/fees summary CSV before it enters the paper table."""

    required = {"cohort", "block_time", *SUMMARY_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source_csv} is missing required columns: {missing}")

    cohorts = set(df["cohort"].astype(str))
    if cohorts != set(COHORTS):
        raise ValueError(f"{source_csv} has cohorts {sorted(cohorts)}, expected {sorted(COHORTS)}")

    blocks = sorted(df["block_time"].astype(int).unique().tolist())
    if blocks != list(range(2, 17)):
        raise ValueError(f"{source_csv} has block_time values {blocks}, expected 2..16")

    counts = df.groupby(["cohort", "block_time"]).size()
    if not (counts == 1).all():
        repeated = counts[counts != 1]
        raise ValueError(f"{source_csv} has non-unique cohort/block_time rows for {schedule}: {repeated.to_dict()}")


def rebuild_data_from_source_csvs(source_root: Path, output_path: Path) -> pd.DataFrame:
    """Rebuild the long-form paper ratio table from exact production summaries."""

    rows: list[pd.DataFrame] = []
    for schedule, cfg in SCHEDULES.items():
        source_csv = source_root / cfg["source_csv"]
        if not source_csv.exists():
            raise FileNotFoundError(f"Missing source summary for {schedule}: {source_csv}")
        df = pd.read_csv(source_csv)
        _validate_summary(df, source_csv=source_csv, schedule=schedule)
        df = df.copy()
        df["block_time"] = df["block_time"].astype(int)
        df["cohort"] = df["cohort"].astype(str)
        for column in SUMMARY_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="raise")
        df["fee_schedule"] = schedule
        df["fee_schedule_label"] = str(cfg["label"])
        df["cohort_label"] = df["cohort"].map(COHORTS)
        df["median_R"] = df["median"]
        df["source"] = str(source_csv)
        rows.append(
            df[
                [
                    "block_time",
                    "fee_schedule",
                    "fee_schedule_label",
                    "cohort",
                    "cohort_label",
                    "n",
                    "mean",
                    "std",
                    "median_R",
                    "p2_5",
                    "p25",
                    "p75",
                    "p97_5",
                    "source",
                ]
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    schedule_order = {name: idx for idx, name in enumerate(SCHEDULES)}
    cohort_order = {name: idx for idx, name in enumerate(COHORTS)}
    out = out.sort_values(
        by=["block_time", "fee_schedule", "cohort"],
        key=lambda series: series.map(schedule_order)
        if series.name == "fee_schedule"
        else series.map(cohort_order)
        if series.name == "cohort"
        else series,
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def build_figure(df: pd.DataFrame, style: Mapping[str, Any]) -> go.Figure:
    """Build a three-panel line plot of median R_t by block size."""
    font = dict(style["font"])
    line_cfg = dict(style["line"])
    margins = dict(style["margins"])
    legend_cfg = dict(style.get("legend", {}))
    panel_cfg = dict(style.get("panel", {}))

    fig = make_subplots(
        rows=1,
        cols=len(COHORTS),
        subplot_titles=[COHORTS[key] for key in COHORTS],
        shared_yaxes=False,
        horizontal_spacing=float(panel_cfg.get("horizontal_spacing", 0.08)),
    )

    for col_idx, cohort in enumerate(COHORTS, start=1):
        cohort_df = df[df["cohort"] == cohort]
        for schedule in SCHEDULES:
            cfg = _schedule_config(schedule, style)
            plot_df = cohort_df[cohort_df["fee_schedule"] == schedule].sort_values("block_time")
            fig.add_trace(
                go.Scatter(
                    x=plot_df["block_time"],
                    y=plot_df["median_R"],
                    mode="lines+markers",
                    name=str(cfg["label"]),
                    legendgroup=schedule,
                    showlegend=(col_idx == 1 and bool(legend_cfg.get("show", True))),
                    line=dict(
                        color=str(cfg["color"]),
                        width=float(line_cfg.get("width", 4)),
                        dash=str(cfg["dash"]),
                    ),
                    marker=dict(
                        color=str(cfg["color"]),
                        size=float(line_cfg.get("marker_size", 9)),
                    ),
                    hovertemplate=(
                        f"{COHORTS[cohort]}<br>"
                        f"{cfg['label']}<br>"
                        "B=%{x}<br>median R=%{y:.3g}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        # R=1 is the economically relevant marginal fee-coverage threshold.
        fig.add_hline(
            y=1.0,
            line=dict(color="#555555", width=1.5, dash="dash"),
            row=1,
            col=col_idx,
        )
        fig.add_hline(
            y=0.0,
            line=dict(color="#aaaaaa", width=1.0, dash="dot"),
            row=1,
            col=col_idx,
        )
        fig.update_xaxes(
            title_text="Block size B",
            title_font=dict(size=int(font["axis_title_size"])),
            tickfont=dict(size=int(font["tick_size"])),
            dtick=2,
            range=[1.6, 16.4],
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            title_text="Median Rₜ = ΔLVR / ΔFees" if col_idx == 1 else "",
            title_font=dict(size=int(font["axis_title_size"])),
            tickfont=dict(size=int(font["tick_size"])),
            zeroline=False,
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        template=str(style["template"]),
        width=int(style["width"]),
        height=int(style["height"]),
        margin=dict(l=int(margins["l"]), r=int(margins["r"]), t=int(margins["t"]), b=int(margins["b"])),
        font=dict(
            family=str(font["family"]),
            size=int(font["base_size"]),
            color="black",
        ),
        legend=dict(
            orientation=str(legend_cfg.get("orientation", "h")),
            yanchor=str(legend_cfg.get("yanchor", "bottom")),
            y=float(legend_cfg.get("y", 1.08)),
            xanchor=str(legend_cfg.get("xanchor", "center")),
            x=float(legend_cfg.get("x", 0.5)),
            font=dict(size=int(font["legend_size"])),
        ),
        showlegend=bool(legend_cfg.get("show", True)),
    )
    fig.update_annotations(font=dict(size=int(font["subplot_title_size"]), color="black"))
    return fig


def write_outputs(fig: go.Figure, output_path: Path, *, scale: float) -> Dict[str, str]:
    """Write HTML, PNG, and PDF sidecars for the combined figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_path.with_suffix(".html")
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")

    fig.write_html(html_path, include_plotlyjs="cdn")
    fig.write_image(png_path, width=int(fig.layout.width), height=int(fig.layout.height), scale=float(scale))
    fig.write_image(pdf_path, width=int(fig.layout.width), height=int(fig.layout.height), scale=float(scale))
    return {"html": str(html_path), "png": str(png_path), "pdf": str(pdf_path)}


def write_manifest(
    *,
    data_path: Path,
    style_path: Path,
    source_root: Path,
    output_paths: Mapping[str, str],
    manifest_path: Path,
) -> None:
    """Persist minimal provenance for the generated paper asset."""

    def _manifest_path(path: Path | str) -> str:
        candidate = Path(path).resolve()
        try:
            return str(candidate.relative_to(_REPO_ROOT))
        except ValueError:
            return str(candidate)

    manifest = {
        "script": str(Path(__file__).relative_to(_REPO_ROOT)),
        "data": _manifest_path(data_path),
        "style_config": _manifest_path(style_path),
        "source_root": str(source_root),
        "source_summaries": {
            schedule: str(source_root / cfg["source_csv"])
            for schedule, cfg in SCHEDULES.items()
        },
        "outputs": {key: _manifest_path(value) for key, value in output_paths.items()},
        "source_note": (
            "All plotted R_t medians are read from exact production Model 2 "
            "dLVR_over_dFees_summary.csv files under the configured source root. "
            "No PNG digitization is used for this figure when --rebuild-data-from-source-csv is used."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--style-config", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--rebuild-data-from-source-csv",
        action="store_true",
        help=(
            "Rebuild the companion CSV from exact /mnt/external/scenarios "
            "dLVR_over_dFees_summary.csv files before plotting."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "paper" / "tables" / "model2_blocksize_ratio_combined_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    style_path = args.style_config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()

    style = load_paper_style(style_path)
    if args.rebuild_data_from_source_csv:
        df = rebuild_data_from_source_csvs(source_root, data_path)
        print(f"wrote {data_path}")
    else:
        df = load_ratio_data(data_path)
    fig = build_figure(df, style)
    output_paths = write_outputs(fig, output_path, scale=float(style.get("scale", 1)))
    write_manifest(
        data_path=data_path,
        style_path=style_path,
        source_root=source_root,
        output_paths=output_paths,
        manifest_path=manifest_path,
    )

    print("wrote " + output_paths["png"])
    print("wrote " + output_paths["pdf"])
    print("wrote " + output_paths["html"])
    print("wrote " + str(manifest_path))


if __name__ == "__main__":
    main()
