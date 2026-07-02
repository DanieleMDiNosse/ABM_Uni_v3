#!/usr/bin/env python3
"""Regenerate all active figures referenced by ``paper/ABM_paper.tex``.

The script is intentionally cache-first.  It rebuilds the current paper figure
snapshots from existing machine-readable artifacts where possible and only
requires expensive simulator reruns for artifacts whose cached data are absent.
It scans the configured external roots (by default under ``/mnt/external``) for
block-size summaries and backup paper sidecars before falling back to local
paper tables.

Default active figures covered
------------------------------
- ``images/259572_static_LPpassiveshare1_pjit0.0_1_price_steps10000.png``
- ``images/microstructure_price_static_vs_toxicity.pdf``
- ``images/microstructure_acf_static_vs_toxicity.pdf``
- ``images/analysis/png/pnl_heatmap.png``
- ``images/model2_blocksize_ratio_combined.pdf``
- ``images/model2_delta_lvr_blocksize_combined.pdf``

Examples
--------
Regenerate every active paper figure from cached data::

    conda run -n main python -m scripts.analysis.regenerate_paper_figures

Audit coverage without writing files::

    conda run -n main python -m scripts.analysis.regenerate_paper_figures --dry-run

Write outputs to temporary folders for a smoke test::

    conda run -n main python -m scripts.analysis.regenerate_paper_figures \
        --image-dir /tmp/abm-paper-images --table-dir /tmp/abm-paper-tables
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis import plot_model2_delta_lvr_combined as delta_lvr  # noqa: E402
from scripts.analysis import plot_model2_lvr_ratio_combined as lvr_ratio  # noqa: E402
from scripts.analysis.paper_figure_style import load_figure_style  # noqa: E402

PAPER_TEX = _REPO_ROOT / "paper" / "ABM_paper.tex"
PAPER_DIR = _REPO_ROOT / "paper"
DEFAULT_IMAGE_DIR = PAPER_DIR / "images"
DEFAULT_TABLE_DIR = PAPER_DIR / "tables"
DEFAULT_SOURCE_ROOTS = [
    Path("/mnt/external/scenarios"),
    Path("/mnt/external/Backup/repositories/ABM_Uni_v3"),
    Path("/mnt/external/Documents/repositories/ABM_Uni_v3"),
    Path("/mnt/external"),
]

EXPECTED_ACTIVE_FIGURES = {
    "images/259572_static_LPpassiveshare1_pjit0.0_1_price_steps10000.png",
    "images/microstructure_price_static_vs_toxicity.pdf",
    "images/microstructure_acf_static_vs_toxicity.pdf",
    "images/analysis/png/pnl_heatmap.png",
    "images/model2_blocksize_ratio_combined.pdf",
    "images/model2_delta_lvr_blocksize_combined.pdf",
}

REPRESENTATIVE_PRICE_NAME = "259572_static_LPpassiveshare1_pjit0.0_1_price_steps10000"
STATIC_DIAGNOSTIC_RUN_REL = Path(
    "abm_results/scenarios/section4_microstructure_diagnostics/runs/static_seed10_T13000"
)

DEFAULT_PLOTLY_FIGURE_STYLE: dict[str, Any] = {
    "template": "plotly_white",
    "width": 1400,
    "height": 900,
    "scale": 1,
    "font": {
        "family": "Arial, sans-serif",
        "base_size": 18,
        "axis_title_size": 20,
        "tick_size": 16,
        "legend_size": 16,
        "subplot_title_size": 20,
    },
    "margins": {"l": 90, "r": 40, "t": 90, "b": 80},
    "legend": {
        "show": True,
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0.0,
    },
    "grid": {"show": True, "color": "#e1e1e1", "width": 1.0},
    "line": {"width": 2.0, "dash_width": 1.6},
}

DEFAULT_MATPLOTLIB_FIGURE_STYLE: dict[str, Any] = {
    "figsize": [12.8, 4.4],
    "dpi": 300,
    "font": {
        "title_size": 13,
        "axis_title_size": 11,
        "tick_size": 10,
        "legend_size": 9,
    },
    "legend": {"show": True, "loc": "upper left", "frameon": True},
    "grid": {"show": True, "color": "#bbbbbb", "alpha": 0.35, "width": 0.8},
    "line": {"cex_width": 1.7, "dex_width": 1.9, "reference_width": 1.0},
}


@dataclass(frozen=True)
class PlannedOutput:
    figure: str
    outputs: tuple[Path, ...]
    sources: tuple[Path, ...]
    note: str


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _style_mapping(style: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = style.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for style key '{key}'")
    return dict(value)


def _style_figsize(style: Mapping[str, Any], *, default: tuple[float, float]) -> tuple[float, float]:
    value = style.get("figsize", default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value) if isinstance(value, Sequence) else []
    if len(parts) != 2:
        raise ValueError(f"Expected two values for figsize, got {value!r}")
    return (float(parts[0]), float(parts[1]))


def _plotly_dimensions(style: Mapping[str, Any]) -> tuple[int, int, float]:
    return int(style.get("width", 1400)), int(style.get("height", 900)), float(style.get("scale", 1))


def _apply_plotly_layout(fig: go.Figure, style: Mapping[str, Any], *, title: str | None = None) -> None:
    font = _style_mapping(style, "font")
    margins = _style_mapping(style, "margins")
    legend = _style_mapping(style, "legend")
    width, height, _scale = _plotly_dimensions(style)
    layout: dict[str, Any] = {
        "template": str(style.get("template", "plotly_white")),
        "width": width,
        "height": height,
        "margin": {
            "l": int(margins.get("l", 90)),
            "r": int(margins.get("r", 40)),
            "t": int(margins.get("t", 90)),
            "b": int(margins.get("b", 80)),
        },
        "font": {
            "family": str(font.get("family", "Arial, sans-serif")),
            "size": int(font.get("base_size", 18)),
            "color": str(font.get("color", "black")),
        },
        "showlegend": bool(legend.get("show", True)),
        "legend": {
            "orientation": str(legend.get("orientation", "h")),
            "yanchor": str(legend.get("yanchor", "bottom")),
            "y": float(legend.get("y", 1.02)),
            "xanchor": str(legend.get("xanchor", "left")),
            "x": float(legend.get("x", 0.0)),
            "font": {"size": int(font.get("legend_size", 16))},
        },
    }
    if title is not None:
        layout["title"] = title
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(size=int(font.get("subplot_title_size", 20)), color="black"))


def _strip_comments(line: str) -> str:
    """Strip unescaped TeX comments from one line."""

    out: list[str] = []
    escaped = False
    for char in line:
        if char == "%" and not escaped:
            break
        out.append(char)
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return "".join(out)


def active_includegraphics(tex_path: Path = PAPER_TEX) -> list[str]:
    """Return active ``\\includegraphics`` paths, excluding comments/``\\iffalse`` blocks."""

    active_paths: list[str] = []
    iffalse_depth = 0
    for raw_line in tex_path.read_text(encoding="utf-8").splitlines():
        line = _strip_comments(raw_line)
        opens_iffalse = bool(re.search(r"\\iffalse\b", line))
        closes_fi = bool(re.search(r"\\fi\b", line))
        if opens_iffalse:
            iffalse_depth += 1
        if iffalse_depth == 0:
            active_paths.extend(
                match.group(1)
                for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", line)
            )
        if iffalse_depth > 0 and closes_fi:
            iffalse_depth -= 1
    return active_paths


def _existing_roots(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    roots: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return roots


def _candidate_data_roots(source_roots: Sequence[Path]) -> list[Path]:
    """Return roots that may contain ``model2_static/...`` style scenario folders."""

    candidates: list[Path] = []
    for root in _existing_roots(source_roots):
        candidates.append(root)
        candidates.append(root / "abm_results" / "scenarios")
        if root.name == "scenarios":
            candidates.append(root)
    return _existing_roots(candidates)


def _find_first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_external_file(source_roots: Sequence[Path], rel_path: Path) -> Path | None:
    """Find *rel_path* directly under common external ABM roots."""

    candidates: list[Path] = []
    for root in _existing_roots(source_roots):
        candidates.extend(
            [
                root / rel_path,
                root / "paper" / rel_path,
                root / "Backup" / "repositories" / "ABM_Uni_v3" / rel_path,
                root / "Backup" / "repositories" / "ABM_Uni_v3" / "paper" / rel_path,
                root / "Documents" / "repositories" / "ABM_Uni_v3" / rel_path,
                root / "Documents" / "repositories" / "ABM_Uni_v3" / "paper" / rel_path,
            ]
        )
    return _find_first_existing(candidates)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _load_jsonish_from_html(html_path: Path) -> tuple[list[Any], Mapping[str, Any]]:
    """Extract Plotly ``data`` and ``layout`` JSON from a ``fig.write_html`` file."""

    text = html_path.read_text(encoding="utf-8")
    marker = "Plotly.newPlot("
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find Plotly.newPlot in {html_path}")
    decoder = json.JSONDecoder()

    data_start = text.find("[", start)
    if data_start < 0:
        raise ValueError(f"Could not find Plotly data array in {html_path}")
    data, data_end = decoder.raw_decode(text[data_start:])
    after_data = data_start + data_end
    layout_start = text.find("{", after_data)
    if layout_start < 0:
        raise ValueError(f"Could not find Plotly layout object in {html_path}")
    layout, _layout_end = decoder.raw_decode(text[layout_start:])
    if not isinstance(data, list) or not isinstance(layout, Mapping):
        raise TypeError(f"Unexpected Plotly HTML payload in {html_path}")
    return data, layout


def _decode_plotly_array(value: Any) -> Any:
    """Decode Plotly typed-array JSON objects emitted by recent Plotly versions."""

    if not isinstance(value, Mapping) or "bdata" not in value:
        return value
    dtype_key = str(value.get("dtype", "f8"))
    dtype_map = {
        "f8": np.float64,
        "f4": np.float32,
        "i4": np.int32,
        "i8": np.int64,
        "u4": np.uint32,
        "u8": np.uint64,
    }
    if dtype_key not in dtype_map:
        raise ValueError(f"Unsupported Plotly typed-array dtype: {dtype_key}")
    arr = np.frombuffer(base64.b64decode(str(value["bdata"])), dtype=dtype_map[dtype_key])
    shape_raw = value.get("shape")
    if shape_raw:
        if isinstance(shape_raw, str):
            shape = tuple(int(part.strip()) for part in shape_raw.split(",") if part.strip())
        else:
            shape = tuple(int(part) for part in shape_raw)
        arr = arr.reshape(shape)
    return arr.tolist()


def _write_plotly_outputs(
    fig: go.Figure,
    output_png: Path,
    *,
    width: int,
    height: int,
    scale: float,
) -> dict[str, Path]:
    """Write PNG plus HTML sidecar; return all paths."""

    output_png.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_png.with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    fig.write_image(output_png, width=width, height=height, scale=scale)
    return {"png": output_png, "html": html_path}


def find_static_run_output_data(source_roots: Sequence[Path]) -> Path | None:
    """Find cached static diagnostic ``output_data`` arrays."""

    candidates = [
        _REPO_ROOT / STATIC_DIAGNOSTIC_RUN_REL / "output_data",
    ]
    scenario_run_rel = STATIC_DIAGNOSTIC_RUN_REL.relative_to("abm_results/scenarios")
    for root in _existing_roots(source_roots):
        candidates.extend(
            [
                root / STATIC_DIAGNOSTIC_RUN_REL / "output_data",
                root / "abm_results" / "scenarios" / scenario_run_rel / "output_data",
                root / "Backup" / "repositories" / "ABM_Uni_v3" / STATIC_DIAGNOSTIC_RUN_REL / "output_data",
                root
                / "Backup"
                / "repositories"
                / "ABM_Uni_v3"
                / "abm_results"
                / "scenarios"
                / scenario_run_rel
                / "output_data",
            ]
        )
    for candidate in candidates:
        if (candidate / "dex_price_end_of_block.npy").exists() and (
            candidate / "cex_dex_spread_end_of_block.npy"
        ).exists():
            return candidate
    return None


def regenerate_representative_static_price(
    *,
    image_dir: Path,
    table_dir: Path,
    style: Mapping[str, Any],
    source_roots: Sequence[Path],
    dry_run: bool,
) -> PlannedOutput:
    """Regenerate the full static CEX-vs-DEX price diagnostic from cached arrays."""

    output_png = image_dir / f"{REPRESENTATIVE_PRICE_NAME}.png"
    table_path = table_dir / "representative_static_price_values.csv"
    source = find_static_run_output_data(source_roots)
    if source is None:
        backup = _find_external_file(
            source_roots,
            Path("paper/images") / f"{REPRESENTATIVE_PRICE_NAME}.png",
        )
        if backup is None:
            raise FileNotFoundError(
                "Could not find cached static diagnostic arrays or backup figure for the representative price panel."
            )
        if not dry_run:
            output_png.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, output_png)
        return PlannedOutput(
            figure="representative_static_price",
            outputs=(output_png,),
            sources=(backup,),
            note="Copied backup image because raw cached arrays were unavailable.",
        )

    if dry_run:
        return PlannedOutput(
            figure="representative_static_price",
            outputs=(output_png, table_path, output_png.with_suffix(".html")),
            sources=(source,),
            note="Would rebuild price panel from cached static diagnostic arrays.",
        )

    dex = np.load(source / "dex_price_end_of_block.npy").astype(float)
    spread = np.load(source / "cex_dex_spread_end_of_block.npy").astype(float)
    n = min(dex.size, spread.size)
    dex = dex[:n]
    # Repository tests define the signed spread as DEX - CEX.
    cex = dex - spread[:n]
    fee = 1.0e-4
    flash = 1.0e-4
    band_lo = cex * (1.0 - fee) / (1.0 + flash)
    band_hi = cex * (1.0 + flash) / (1.0 - fee)
    skip_candidate = max(0, int(style.get("skip_blocks", 3000)))
    skip = skip_candidate if n > skip_candidate else 0
    steps = np.arange(n, dtype=int)
    view = slice(skip, n)
    steps_v = steps[view]
    dex_v = dex[view]
    cex_v = cex[view]
    band_lo_v = band_lo[view]
    band_hi_v = band_hi[view]

    rows = (
        {
            "block": int(step),
            "cex_price": float(m),
            "dex_price": float(p),
            "band_low": float(lo),
            "band_high": float(hi),
            "source_output_data": str(source),
        }
        for step, m, p, lo, hi in zip(steps_v, cex_v, dex_v, band_lo_v, band_hi_v)
    )
    _write_csv(
        table_path,
        rows,
        ["block", "cex_price", "dex_price", "band_low", "band_high", "source_output_data"],
    )

    def _log_returns(prices: np.ndarray) -> np.ndarray:
        mask = np.isfinite(prices) & (prices > 0)
        clean = prices[mask]
        return np.diff(np.log(clean)) if clean.size >= 2 else np.array([], dtype=float)

    font = _style_mapping(style, "font")
    colors = _style_mapping(style, "colors")
    grid = _style_mapping(style, "grid")
    line_cfg = _style_mapping(style, "line")
    histogram_cfg = _style_mapping(style, "histogram")
    panel_cfg = _style_mapping(style, "panel")

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        column_widths=list(panel_cfg.get("column_widths", [0.72, 0.28])),
        horizontal_spacing=float(panel_cfg.get("horizontal_spacing", 0.08)),
        vertical_spacing=float(panel_cfg.get("vertical_spacing", 0.12)),
        subplot_titles=tuple(panel_cfg.get("subplot_titles", ["", "CEX log-returns", "DEX log-returns", ""])),
    )
    fig.add_trace(
        go.Scatter(x=steps_v, y=band_lo_v, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps_v,
            y=band_hi_v,
            mode="lines",
            fill="tonexty",
            fillcolor=str(colors.get("no_arb_band_fill", "rgba(180,180,180,0.35)")),
            line=dict(width=0),
            name="No-arb fee band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps_v,
            y=dex_v,
            mode="lines",
            name="DEX price Pₜ",
            line=dict(color=str(colors.get("dex_price", "#00b894")), width=float(line_cfg.get("dex_width", 2.0))),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps_v,
            y=cex_v,
            mode="lines",
            name="CEX price mₜ",
            line=dict(
                color=str(colors.get("cex_price", "#8b5cf6")),
                width=float(line_cfg.get("cex_width", 1.6)),
                dash=str(line_cfg.get("cex_dash", "dash")),
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=_log_returns(cex_v),
            nbinsx=int(histogram_cfg.get("bins", 60)),
            marker_color=str(colors.get("cex_returns", "#1f77b4")),
            opacity=float(histogram_cfg.get("opacity", 0.85)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=_log_returns(dex_v),
            nbinsx=int(histogram_cfg.get("bins", 60)),
            marker_color=str(colors.get("dex_returns", "#ff7f0e")),
            opacity=float(histogram_cfg.get("opacity", 0.85)),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    _apply_plotly_layout(fig, style, title=str(style.get("title", "CEX vs DEX Price")))
    axis_title = dict(size=int(font.get("axis_title_size", 20)))
    tickfont = dict(size=int(font.get("tick_size", 16)))
    fig.update_xaxes(title_text="Block", title_font=axis_title, tickfont=tickfont, row=1, col=1)
    fig.update_yaxes(title_text="Price", title_font=axis_title, tickfont=tickfont, row=1, col=1)
    fig.update_xaxes(title_text="Log-return", title_font=axis_title, tickfont=tickfont, row=1, col=2)
    fig.update_xaxes(title_text="Log-return", title_font=axis_title, tickfont=tickfont, row=2, col=2)
    fig.update_yaxes(title_text="Count", type="log", title_font=axis_title, tickfont=tickfont, row=1, col=2)
    fig.update_yaxes(title_text="Count", type="log", title_font=axis_title, tickfont=tickfont, row=2, col=2)
    if bool(grid.get("show", True)):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=str(grid.get("color", "#e1e1e1")),
            gridwidth=float(grid.get("width", 1)),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=str(grid.get("color", "#e1e1e1")),
            gridwidth=float(grid.get("width", 1)),
        )
    width, height, scale = _plotly_dimensions(style)
    outputs = _write_plotly_outputs(fig, output_png, width=width, height=height, scale=scale)
    return PlannedOutput(
        figure="representative_static_price",
        outputs=(outputs["png"], outputs["html"], table_path),
        sources=(source,),
        note="Rebuilt from cached static diagnostic arrays; CEX reconstructed as DEX minus signed spread.",
    )


def _resolve_table(table_dir: Path, source_roots: Sequence[Path], filename: str) -> Path:
    local = table_dir / filename
    if local.exists():
        return local
    default_local = DEFAULT_TABLE_DIR / filename
    if default_local.exists():
        return default_local
    external = _find_external_file(source_roots, Path("paper/tables") / filename)
    if external is not None:
        return external
    raise FileNotFoundError(
        f"Missing table {filename} under {table_dir}, {DEFAULT_TABLE_DIR}, or configured external roots"
    )


def _resolve_optional_table(table_dir: Path, source_roots: Sequence[Path], filename: str) -> Path | None:
    """Resolve an optional provenance table without failing the figure rebuild."""

    try:
        return _resolve_table(table_dir, source_roots, filename)
    except FileNotFoundError:
        return None


def _acf_no_correlation_band(sample_size: int, confidence_level: float = 0.95) -> float:
    """Return the analytic white-noise no-correlation band half-width.

    For a serially uncorrelated return series, the large-sample standard error of
    the sample autocorrelation at any fixed non-zero lag is approximately
    ``1 / sqrt(N)``, where ``N`` is the number of returns. The plotted two-sided
    band is therefore ``z_{1-alpha/2} / sqrt(N)``. This intentionally avoids any
    bootstrap or permutation resampling.
    """

    n = int(sample_size)
    if n <= 0:
        raise ValueError(f"ACF no-correlation band requires positive sample size, got {sample_size}")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")
    from statistics import NormalDist

    z_value = NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)
    return float(z_value / np.sqrt(n))


def _acf_sample_sizes_from_summary(
    table_dir: Path,
    source_roots: Sequence[Path],
    fee_modes: Iterable[str],
) -> dict[str, int]:
    """Return ACF return-sample counts by fee mode from the diagnostics summary table."""

    summary_path = _resolve_optional_table(table_dir, source_roots, "microstructure_fee_diagnostics_values.csv")
    if summary_path is None:
        return {}
    summary_df = pd.read_csv(summary_path)
    required = {"fee_mode", "T", "skip_step"}
    if not required.issubset(summary_df.columns):
        return {}
    wanted = {str(mode) for mode in fee_modes}
    sample_sizes: dict[str, int] = {}
    for row in summary_df.to_dict(orient="records"):
        fee_mode = str(row["fee_mode"])
        if fee_mode not in wanted:
            continue
        # ACF is computed on log returns after skipping the first skip_step
        # price observations, hence N_returns = (T - skip_step) - 1.
        n_returns = int(row["T"]) - int(row["skip_step"]) - 1
        if n_returns > 0:
            sample_sizes[fee_mode] = n_returns
    return sample_sizes


def regenerate_microstructure_figures(
    *,
    image_dir: Path,
    table_dir: Path,
    price_style: Mapping[str, Any],
    acf_style: Mapping[str, Any],
    source_roots: Sequence[Path],
    dry_run: bool,
) -> PlannedOutput:
    """Regenerate the static-vs-toxicity zoom and ACF figures from CSV tables."""

    price_csv = _resolve_table(table_dir, source_roots, "microstructure_price_zoom_values.csv")
    acf_csv = _resolve_table(table_dir, source_roots, "microstructure_acf_values.csv")
    price_png = image_dir / "microstructure_price_static_vs_toxicity.png"
    price_pdf = image_dir / "microstructure_price_static_vs_toxicity.pdf"
    acf_png = image_dir / "microstructure_acf_static_vs_toxicity.png"
    acf_pdf = image_dir / "microstructure_acf_static_vs_toxicity.pdf"
    if dry_run:
        return PlannedOutput(
            figure="microstructure_static_vs_toxicity",
            outputs=(price_png, price_pdf, acf_png, acf_pdf),
            sources=(price_csv, acf_csv),
            note="Would rebuild microstructure comparison figures from stored CSV tables.",
        )

    import matplotlib.pyplot as plt

    price_df = pd.read_csv(price_csv)
    acf_df = pd.read_csv(acf_csv)
    required_price = {"fee_mode", "block", "cex_price", "dex_price", "band_low", "band_high"}
    required_acf = {"fee_mode", "lag", "acf"}
    if missing := sorted(required_price - set(price_df.columns)):
        raise ValueError(f"{price_csv} missing columns: {missing}")
    if missing := sorted(required_acf - set(acf_df.columns)):
        raise ValueError(f"{acf_csv} missing columns: {missing}")

    price_png.parent.mkdir(parents=True, exist_ok=True)
    titles = {"static": "Static fees (narrow band)", "toxicity": "Toxicity fees (wider band)"}
    price_font = _style_mapping(price_style, "font")
    price_colors = _style_mapping(price_style, "colors")
    price_line = _style_mapping(price_style, "line")
    price_grid = _style_mapping(price_style, "grid")
    price_legend = _style_mapping(price_style, "legend")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=_style_figsize(price_style, default=(12.8, 4.4)),
        sharex=True,
        sharey=False,
    )
    for ax, fee_mode in zip(axes, ["static", "toxicity"]):
        df = price_df[price_df["fee_mode"] == fee_mode].sort_values("block")
        ax.fill_between(
            df["block"],
            df["band_low"],
            df["band_high"],
            color=str(price_colors.get("no_arb_band", "#d9d9d9")),
            alpha=float(price_colors.get("no_arb_band_alpha", 0.85)),
            label="No-arb fee band",
        )
        ax.plot(
            df["block"],
            df["cex_price"],
            color=str(price_colors.get("cex_price", "#8b5cf6")),
            linestyle=str(price_line.get("cex_style", "--")),
            linewidth=float(price_line.get("cex_width", 1.7)),
            label="CEX price $m_t$",
        )
        ax.plot(
            df["block"],
            df["dex_price"],
            color=str(price_colors.get("dex_price", "#00b894")),
            linewidth=float(price_line.get("dex_width", 1.9)),
            label="DEX price $P_t$",
        )
        ax.set_title(titles[fee_mode], fontsize=int(price_font.get("title_size", 13)))
        ax.set_xlabel("Block", fontsize=int(price_font.get("axis_title_size", 11)))
        ax.grid(
            bool(price_grid.get("show", True)),
            color=str(price_grid.get("color", "#bbbbbb")),
            alpha=float(price_grid.get("alpha", 0.35)),
            linewidth=float(price_grid.get("width", 0.8)),
        )
        ax.tick_params(axis="both", labelsize=int(price_font.get("tick_size", 10)))
    axes[0].set_ylabel("Price", fontsize=int(price_font.get("axis_title_size", 11)))
    if bool(price_legend.get("show", True)):
        axes[0].legend(
            loc=str(price_legend.get("loc", "upper left")),
            fontsize=int(price_font.get("legend_size", 9)),
            frameon=bool(price_legend.get("frameon", True)),
        )
    fig.tight_layout()
    fig.savefig(price_png, dpi=int(price_style.get("dpi", 300)), bbox_inches="tight")
    fig.savefig(price_pdf, bbox_inches="tight")
    plt.close(fig)

    acf_font = _style_mapping(acf_style, "font")
    acf_colors = _style_mapping(acf_style, "colors")
    acf_line = _style_mapping(acf_style, "line")
    acf_grid = _style_mapping(acf_style, "grid")
    acf_legend = _style_mapping(acf_style, "legend")
    acf_band_cfg = _style_mapping(acf_style, "no_correlation_band")
    confidence_level = float(acf_band_cfg.get("confidence_level", 0.95))
    fee_modes = [str(mode) for mode in acf_df["fee_mode"].dropna().unique()]
    acf_sample_sizes = _acf_sample_sizes_from_summary(table_dir, source_roots, fee_modes)
    if bool(acf_band_cfg.get("show", True)) and not acf_sample_sizes:
        raise ValueError(
            "Cannot draw analytic ACF no-correlation bands because "
            "microstructure_fee_diagnostics_values.csv with T and skip_step was not found."
        )
    acf_bands = {
        fee_mode: _acf_no_correlation_band(sample_size, confidence_level)
        for fee_mode, sample_size in acf_sample_sizes.items()
    }
    y_min = float(acf_df["acf"].min())
    y_max = float(acf_df["acf"].max())
    if acf_bands:
        max_band = max(acf_bands.values())
        y_min = min(y_min, -max_band)
        y_max = max(y_max, max_band)
    y_padding = float(acf_style.get("y_padding", 0.04))
    y_limits = (min(-0.02, y_min - y_padding), max(0.04, y_max + y_padding))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=_style_figsize(acf_style, default=(11.2, 4.0)),
        sharex=True,
        sharey=True,
    )
    max_lag = int(acf_df["lag"].max())
    for ax, fee_mode in zip(axes, ["static", "toxicity"]):
        df = acf_df[acf_df["fee_mode"] == fee_mode].sort_values("lag")
        lag1 = float(df.loc[df["lag"] == 1, "acf"].iloc[0])
        ax.bar(
            df["lag"],
            df["acf"],
            color=str(acf_colors.get("acf_bar", "#636EFA")),
            width=float(acf_style.get("bar_width", 0.78)),
        )
        band = acf_bands.get(fee_mode)
        if bool(acf_band_cfg.get("show", True)) and band is not None:
            band_label = (
                f"{confidence_level:.0%} no-correlation band"
                if confidence_level >= 0.1
                else "No-correlation band"
            )
            ax.axhspan(
                -band,
                band,
                color=str(acf_band_cfg.get("color", "#999999")),
                alpha=float(acf_band_cfg.get("alpha", 0.16)),
                label=band_label,
                zorder=0,
            )
            ax.axhline(
                band,
                color=str(acf_band_cfg.get("edge_color", acf_band_cfg.get("color", "#999999"))),
                linestyle=str(acf_band_cfg.get("edge_style", ":")),
                linewidth=float(acf_band_cfg.get("edge_width", 1.0)),
                zorder=1,
            )
            ax.axhline(
                -band,
                color=str(acf_band_cfg.get("edge_color", acf_band_cfg.get("color", "#999999"))),
                linestyle=str(acf_band_cfg.get("edge_style", ":")),
                linewidth=float(acf_band_cfg.get("edge_width", 1.0)),
                zorder=1,
            )
        ax.axhline(
            0.0,
            color=str(acf_colors.get("zero_line", "#555555")),
            linestyle=str(acf_line.get("reference_style", "--")),
            linewidth=float(acf_line.get("reference_width", 1.0)),
        )
        ax.set_title(f"{fee_mode.capitalize()} fee (lag 1 = {lag1:.3f})", fontsize=int(acf_font.get("title_size", 13)))
        ax.set_xlabel("Lag (blocks)", fontsize=int(acf_font.get("axis_title_size", 11)))
        ax.set_xlim(0.4, max_lag + 0.6)
        ax.set_ylim(*y_limits)
        ax.grid(
            bool(acf_grid.get("show", True)),
            axis="y",
            color=str(acf_grid.get("color", "#bbbbbb")),
            alpha=float(acf_grid.get("alpha", 0.35)),
            linewidth=float(acf_grid.get("width", 0.8)),
        )
        ax.tick_params(axis="both", labelsize=int(acf_font.get("tick_size", 10)))
    axes[0].set_ylabel("Autocorrelation", fontsize=int(acf_font.get("axis_title_size", 11)))
    if bool(acf_legend.get("show", True)) and bool(acf_band_cfg.get("show", True)):
        axes[0].legend(
            loc=str(acf_legend.get("loc", "lower right")),
            fontsize=int(acf_font.get("legend_size", 9)),
            frameon=bool(acf_legend.get("frameon", True)),
        )
    fig.tight_layout()
    fig.savefig(acf_png, dpi=int(acf_style.get("dpi", 300)), bbox_inches="tight")
    fig.savefig(acf_pdf, bbox_inches="tight")
    plt.close(fig)

    return PlannedOutput(
        figure="microstructure_static_vs_toxicity",
        outputs=(price_png, price_pdf, acf_png, acf_pdf),
        sources=(price_csv, acf_csv),
        note="Rebuilt from microstructure CSV tables; no simulator rerun used.",
    )


def regenerate_pnl_heatmap(
    *,
    image_dir: Path,
    table_dir: Path,
    style: Mapping[str, Any],
    source_roots: Sequence[Path],
    dry_run: bool,
) -> PlannedOutput:
    """Regenerate ``pnl_heatmap.png`` from the existing Plotly HTML sidecar."""

    output_png = image_dir / "analysis" / "png" / "pnl_heatmap.png"
    output_html = image_dir / "analysis" / "html" / "pnl_heatmap.html"
    table_path = table_dir / "pnl_heatmap_values.csv"
    local_html = DEFAULT_IMAGE_DIR / "analysis" / "html" / "pnl_heatmap.html"
    source_html = local_html if local_html.exists() else None
    if source_html is None:
        source_html = _find_external_file(source_roots, Path("paper/images/analysis/html/pnl_heatmap.html"))
    source_csv_fallback = source_html is None and table_path.exists()
    if source_html is None and not source_csv_fallback:
        raise FileNotFoundError(
            "Missing pnl_heatmap HTML sidecar and companion CSV. Re-run scripts.analysis.run_paper_figures "
            "or provide paper/tables/pnl_heatmap_values.csv."
        )
    if dry_run:
        sources: tuple[Path, ...] = (source_html,) if source_html is not None else (table_path,)
        return PlannedOutput(
            figure="pnl_heatmap",
            outputs=(output_png, output_html, table_path),
            sources=sources,
            note="Would rebuild PnL heatmap from existing Plotly HTML sidecar."
            if source_html is not None
            else "Would rebuild PnL heatmap from companion CSV because the HTML sidecar is unavailable.",
        )

    rows: list[dict[str, Any]] = []
    if source_html is not None:
        data, layout = _load_jsonish_from_html(source_html)
        if not data:
            raise ValueError(f"No Plotly traces found in {source_html}")
        trace = dict(data[0])
        x_labels = list(trace.get("x", []))
        y_labels = list(trace.get("y", []))
        z_values = _decode_plotly_array(trace.get("z", []))
        trace["z"] = z_values
        data[0] = trace
        annotations_by_xy: dict[tuple[str, str], str] = {}
        for annotation in layout.get("annotations", []):
            x = str(annotation.get("x", ""))
            y = str(annotation.get("y", ""))
            annotations_by_xy[(x, y)] = str(annotation.get("text", ""))
        for i, cohort in enumerate(y_labels):
            for j, scenario in enumerate(x_labels):
                value = z_values[i][j]
                rows.append(
                    {
                        "scenario": scenario,
                        "cohort": cohort,
                        "mean_hedged_pnl": value,
                        "annotation": annotations_by_xy.get((scenario, cohort), ""),
                        "source_html": str(source_html),
                    }
                )
        _write_csv(table_path, rows, ["scenario", "cohort", "mean_hedged_pnl", "annotation", "source_html"])
        fig = go.Figure(data=data, layout=layout)
    else:
        csv_df = pd.read_csv(table_path)
        required = {"scenario", "cohort", "mean_hedged_pnl", "annotation"}
        if missing := sorted(required - set(csv_df.columns)):
            raise ValueError(f"{table_path} missing columns required for heatmap fallback: {missing}")
        x_labels = list(dict.fromkeys(str(value) for value in csv_df["scenario"].tolist()))
        y_labels = list(dict.fromkeys(str(value) for value in csv_df["cohort"].tolist()))
        z_values: list[list[float | None]] = []
        annotations_by_xy: dict[tuple[str, str], str] = {}
        value_by_xy: dict[tuple[str, str], float | None] = {}
        for row in csv_df.to_dict(orient="records"):
            xy = (str(row["scenario"]), str(row["cohort"]))
            value = row["mean_hedged_pnl"]
            value_by_xy[xy] = None if pd.isna(value) else float(value)
            annotation = row.get("annotation", "")
            annotations_by_xy[xy] = "" if pd.isna(annotation) else str(annotation)
        for cohort in y_labels:
            z_values.append([value_by_xy.get((scenario, cohort)) for scenario in x_labels])
        fig = go.Figure(
            data=[
                go.Heatmap(
                    x=x_labels,
                    y=y_labels,
                    z=z_values,
                    hovertemplate="Scenario=%{x}<br>Cohort=%{y}<br>Mean hedged PnL=%{z}<extra></extra>",
                )
            ]
        )
        for cohort in y_labels:
            for scenario in x_labels:
                text = annotations_by_xy.get((scenario, cohort), "")
                if text:
                    fig.add_annotation(x=scenario, y=cohort, text=text, showarrow=False)
    colors = _style_mapping(style, "colors")
    heatmap_cfg = _style_mapping(style, "heatmap")
    if "colorscale" in colors:
        fig.update_traces(colorscale=colors["colorscale"], selector=dict(type="heatmap"))
    if "zmid" in heatmap_cfg:
        fig.update_traces(zmid=float(heatmap_cfg["zmid"]), selector=dict(type="heatmap"))
    if "colorbar_title" in heatmap_cfg:
        fig.update_traces(colorbar=dict(title=str(heatmap_cfg["colorbar_title"])), selector=dict(type="heatmap"))
    annotation_font_size = heatmap_cfg.get("annotation_font_size")
    if annotation_font_size is not None:
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(size=int(annotation_font_size)))
    _apply_plotly_layout(fig, style, title=str(style.get("title", "")) or None)
    width, height, scale = _plotly_dimensions(style)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    fig.write_image(output_png, width=width, height=height, scale=scale)
    sources = (source_html,) if source_html is not None else (table_path,)
    note = (
        "Rebuilt from existing Plotly HTML sidecar and exported companion CSV."
        if source_html is not None
        else "Rebuilt from companion CSV because the Plotly HTML sidecar was unavailable."
    )
    return PlannedOutput(
        figure="pnl_heatmap",
        outputs=(output_png, output_html, table_path),
        sources=sources,
        note=note,
    )


def _select_blocksize_source_root(source_roots: Sequence[Path], schedules: Mapping[str, Mapping[str, Any]]) -> Path:
    candidates = _candidate_data_roots(source_roots)
    for root in candidates:
        if all((root / cfg["source_csv"]).exists() for cfg in schedules.values()):
            return root
    details = "; ".join(str(root) for root in candidates) or "<none>"
    raise FileNotFoundError(f"No source root contains all required Model 2 block-size summary CSVs. Checked: {details}")


def regenerate_blocksize_ratio(
    *,
    image_dir: Path,
    table_dir: Path,
    style_config: Path,
    source_roots: Sequence[Path],
    dry_run: bool,
) -> PlannedOutput:
    output = image_dir / "model2_blocksize_ratio_combined.png"
    data_path = table_dir / "model2_blocksize_ratio_values.csv"
    manifest = table_dir / "model2_blocksize_ratio_combined_manifest.json"
    source_root = _select_blocksize_source_root(source_roots, lvr_ratio.SCHEDULES)
    if dry_run:
        return PlannedOutput(
            figure="model2_blocksize_ratio_combined",
            outputs=(output, output.with_suffix(".pdf"), output.with_suffix(".html"), data_path, manifest),
            sources=tuple(source_root / cfg["source_csv"] for cfg in lvr_ratio.SCHEDULES.values()),
            note="Would rebuild exact ratio companion CSV from external summaries and regenerate PNG/PDF/HTML.",
        )
    style = lvr_ratio.load_paper_style(style_config)
    rebuilt_from_source = True
    try:
        df = lvr_ratio.rebuild_data_from_source_csvs(source_root, data_path)
    except OSError as exc:
        if not data_path.exists():
            raise
        print(
            f"[model2_blocksize_ratio_combined] Warning: could not read external summaries ({exc}); "
            f"reusing existing companion CSV {data_path}",
            file=sys.stderr,
        )
        df = lvr_ratio.load_ratio_data(data_path)
        rebuilt_from_source = False
    fig = lvr_ratio.build_figure(df, style)
    outputs = lvr_ratio.write_outputs(fig, output, scale=float(style.get("scale", 1)))
    lvr_ratio.write_manifest(
        data_path=data_path.resolve(),
        style_path=style_config.resolve(),
        source_root=source_root,
        output_paths=outputs,
        manifest_path=manifest,
    )
    return PlannedOutput(
        figure="model2_blocksize_ratio_combined",
        outputs=(Path(outputs["png"]), Path(outputs["pdf"]), Path(outputs["html"]), data_path, manifest),
        sources=tuple(source_root / cfg["source_csv"] for cfg in lvr_ratio.SCHEDULES.values())
        if rebuilt_from_source
        else (data_path,),
        note="Rebuilt from exact external dLVR_over_dFees_summary.csv files."
        if rebuilt_from_source
        else "Rebuilt from existing companion CSV because external summary reads failed.",
    )


def regenerate_delta_lvr(
    *,
    image_dir: Path,
    table_dir: Path,
    style_config: Path,
    source_roots: Sequence[Path],
    dry_run: bool,
) -> PlannedOutput:
    output = image_dir / "model2_delta_lvr_blocksize_combined.png"
    data_path = table_dir / "model2_delta_lvr_blocksize_values.csv"
    manifest = table_dir / "model2_delta_lvr_blocksize_combined_manifest.json"
    source_root = _select_blocksize_source_root(source_roots, delta_lvr.SCHEDULES)
    if dry_run:
        return PlannedOutput(
            figure="model2_delta_lvr_blocksize_combined",
            outputs=(output, output.with_suffix(".pdf"), output.with_suffix(".html"), data_path, manifest),
            sources=tuple(source_root / cfg["source_csv"] for cfg in delta_lvr.SCHEDULES.values()),
            note="Would rebuild exact ΔLVR companion CSV from external summaries and regenerate PNG/PDF/HTML.",
        )
    style = delta_lvr.load_paper_style(style_config)
    rebuilt_from_source = True
    try:
        df = delta_lvr.rebuild_data_from_source_csvs(source_root, data_path)
    except OSError as exc:
        if not data_path.exists():
            raise
        print(
            f"[model2_delta_lvr_blocksize_combined] Warning: could not read external summaries ({exc}); "
            f"reusing existing companion CSV {data_path}",
            file=sys.stderr,
        )
        df = delta_lvr.load_delta_lvr_data(data_path)
        rebuilt_from_source = False
    fig = delta_lvr.build_figure(df, style)
    outputs = delta_lvr.write_outputs(fig, output, scale=float(style.get("scale", 1)))
    delta_lvr.write_manifest(
        data_path=data_path.resolve(),
        style_path=style_config.resolve(),
        source_root=source_root,
        output_paths=outputs,
        manifest_path=manifest,
    )
    return PlannedOutput(
        figure="model2_delta_lvr_blocksize_combined",
        outputs=(Path(outputs["png"]), Path(outputs["pdf"]), Path(outputs["html"]), data_path, manifest),
        sources=tuple(source_root / cfg["source_csv"] for cfg in delta_lvr.SCHEDULES.values())
        if rebuilt_from_source
        else (data_path,),
        note="Rebuilt from exact external dLVR_summary.csv files."
        if rebuilt_from_source
        else "Rebuilt from existing companion CSV because external summary reads failed.",
    )


def _copy_external_scenarios_if_missing(
    source_roots: Sequence[Path],
    destination: Path,
    *,
    dry_run: bool,
) -> PlannedOutput | None:
    """Copy the external base scenario YAML into the repo if no local scenario config exists."""

    local = _REPO_ROOT / "configs" / "scenarios" / "section4_microstructure_model0_static.yml"
    if local.exists():
        return None
    source = _find_external_file(source_roots, Path("configs/scenarios/section4_microstructure_model0_static.yml"))
    if source is None:
        source = _find_external_file(source_roots, Path("abm_results/scenarios/test.yml"))
    if source is None:
        source = _find_external_file(source_roots, Path("section4_microstructure_model0_static.yml"))
    if source is None:
        return None
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return PlannedOutput(
        figure="base_scenario_config",
        outputs=(destination,),
        sources=(source,),
        note="Copied external base scenario YAML for optional expensive reruns.",
    )


def maybe_run_full_heatmap_simulation(args: argparse.Namespace) -> PlannedOutput | None:
    """Optional expensive fallback for regenerating heatmap from simulations."""

    if not args.run_heatmap_simulations:
        return None
    config = args.heatmap_config
    if config is None:
        config = _REPO_ROOT / "configs" / "scenarios" / "section4_microstructure_model0_static.yml"
    if not config.exists():
        copied = _copy_external_scenarios_if_missing(args.source_roots, config, dry_run=args.dry_run)
        if copied is None and not config.exists():
            raise FileNotFoundError(
                f"Heatmap simulation config not found: {config}. "
                "Pass --heatmap-config or provide /mnt/external/configs/scenarios/section4_microstructure_model0_static.yml."
            )
        if args.dry_run:
            return copied
    output_dir = args.image_dir / "analysis"
    cmd = [
        sys.executable,
        "-m",
        "scripts.analysis.run_paper_figures",
        "--config",
        str(config),
        "--runs",
        str(args.heatmap_runs),
        "--seed-base",
        str(args.heatmap_seed_base),
        "--max-workers",
        str(args.max_workers),
        "--output-dir",
        str(output_dir),
    ]
    if args.dry_run:
        return PlannedOutput(
            figure="pnl_heatmap_expensive_simulation",
            outputs=(output_dir / "png" / "pnl_heatmap.png", output_dir / "html" / "pnl_heatmap.html"),
            sources=(config,),
            note="Would run expensive multi-seed simulation: " + " ".join(cmd),
        )
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
    return PlannedOutput(
        figure="pnl_heatmap_expensive_simulation",
        outputs=(output_dir / "png" / "pnl_heatmap.png", output_dir / "html" / "pnl_heatmap.html"),
        sources=(config,),
        note="Ran scripts.analysis.run_paper_figures to regenerate heatmap from simulations.",
    )


def validate_active_figures(image_dir: Path) -> dict[str, Any]:
    active = set(active_includegraphics(PAPER_TEX))
    missing_expected = sorted(EXPECTED_ACTIVE_FIGURES - active)
    extra_active = sorted(active - EXPECTED_ACTIVE_FIGURES)
    resolved: dict[str, bool] = {}
    for rel in sorted(active):
        target = image_dir / Path(rel).relative_to("images") if rel.startswith("images/") else PAPER_DIR / rel
        resolved[rel] = target.exists()
    return {
        "active_count": len(active),
        "active_figures": sorted(active),
        "missing_expected": missing_expected,
        "extra_active": extra_active,
        "resolved_under_image_dir": resolved,
    }


def write_overall_manifest(
    path: Path,
    plans: Sequence[PlannedOutput],
    coverage: Mapping[str, Any],
    *,
    style_config: Path,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": str(Path(__file__).relative_to(_REPO_ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "style_config": _repo_rel(style_config),
        "coverage": coverage,
        "plans": [
            {
                "figure": plan.figure,
                "outputs": [_repo_rel(path) for path in plan.outputs],
                "sources": [_repo_rel(path) for path in plan.sources],
                "note": plan.note,
            }
            for plan in plans
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Destination paper image directory.")
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help="Destination paper table/provenance directory.",
    )
    parser.add_argument("--style-config", type=Path, default=PAPER_DIR / "figure_style.yml")
    parser.add_argument(
        "--source-root",
        dest="source_roots",
        action="append",
        type=Path,
        default=None,
        help="External root to scan. Can be supplied multiple times. Defaults to /mnt/external locations.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=["all", "representative-price", "microstructure", "pnl-heatmap", "blocksize-ratio", "delta-lvr"],
        help="Subset of active paper figures to regenerate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List sources and outputs without writing files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_TABLE_DIR / "regenerate_paper_figures_manifest.json",
        help="Overall manifest path.",
    )
    parser.add_argument(
        "--run-heatmap-simulations",
        action="store_true",
        help="Expensive fallback: regenerate pnl_heatmap via multi-seed simulations instead of HTML sidecar.",
    )
    parser.add_argument("--heatmap-config", type=Path, default=None, help="Base config for --run-heatmap-simulations.")
    parser.add_argument("--heatmap-runs", type=int, default=100)
    parser.add_argument("--heatmap-seed-base", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.table_dir = args.table_dir.expanduser().resolve()
    args.style_config = args.style_config.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.source_roots = args.source_roots if args.source_roots is not None else DEFAULT_SOURCE_ROOTS
    args.source_roots = _existing_roots(args.source_roots)

    selected = set(args.figures)
    if "all" in selected:
        selected = {"representative-price", "microstructure", "pnl-heatmap", "blocksize-ratio", "delta-lvr"}

    styles = {
        "representative_static_price": load_figure_style(
            args.style_config,
            DEFAULT_PLOTLY_FIGURE_STYLE,
            figure_key="representative_static_price",
        ),
        "microstructure_price_static_vs_toxicity": load_figure_style(
            args.style_config,
            DEFAULT_MATPLOTLIB_FIGURE_STYLE,
            figure_key="microstructure_price_static_vs_toxicity",
        ),
        "microstructure_acf_static_vs_toxicity": load_figure_style(
            args.style_config,
            DEFAULT_MATPLOTLIB_FIGURE_STYLE,
            figure_key="microstructure_acf_static_vs_toxicity",
        ),
        "pnl_heatmap": load_figure_style(
            args.style_config,
            DEFAULT_PLOTLY_FIGURE_STYLE,
            figure_key="pnl_heatmap",
        ),
    }
    if "blocksize-ratio" in selected:
        lvr_ratio.load_paper_style(args.style_config)
    if "delta-lvr" in selected:
        delta_lvr.load_paper_style(args.style_config)

    coverage = validate_active_figures(args.image_dir)
    if coverage["missing_expected"] or coverage["extra_active"]:
        print("[coverage] Active figure set differs from script mapping:", file=sys.stderr)
        print(json.dumps(coverage, indent=2), file=sys.stderr)

    plans: list[PlannedOutput] = []
    if "representative-price" in selected:
        plans.append(
            regenerate_representative_static_price(
                image_dir=args.image_dir,
                table_dir=args.table_dir,
                style=styles["representative_static_price"],
                source_roots=args.source_roots,
                dry_run=args.dry_run,
            )
        )
    if "microstructure" in selected:
        plans.append(
            regenerate_microstructure_figures(
                image_dir=args.image_dir,
                table_dir=args.table_dir,
                price_style=styles["microstructure_price_static_vs_toxicity"],
                acf_style=styles["microstructure_acf_static_vs_toxicity"],
                source_roots=args.source_roots,
                dry_run=args.dry_run,
            )
        )
    if "pnl-heatmap" in selected:
        if args.run_heatmap_simulations:
            plan = maybe_run_full_heatmap_simulation(args)
            if plan is not None:
                plans.append(plan)
        else:
            plans.append(
                regenerate_pnl_heatmap(
                    image_dir=args.image_dir,
                    table_dir=args.table_dir,
                    style=styles["pnl_heatmap"],
                    source_roots=args.source_roots,
                    dry_run=args.dry_run,
                )
            )
    if "blocksize-ratio" in selected:
        plans.append(
            regenerate_blocksize_ratio(
                image_dir=args.image_dir,
                table_dir=args.table_dir,
                style_config=args.style_config,
                source_roots=args.source_roots,
                dry_run=args.dry_run,
            )
        )
    if "delta-lvr" in selected:
        plans.append(
            regenerate_delta_lvr(
                image_dir=args.image_dir,
                table_dir=args.table_dir,
                style_config=args.style_config,
                source_roots=args.source_roots,
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run:
        write_overall_manifest(args.manifest, plans, coverage, style_config=args.style_config, dry_run=False)
    else:
        # Still print the would-be manifest payload, but do not write it.
        print("[dry-run] overall manifest would be written to", args.manifest)

    for plan in plans:
        print(f"[{plan.figure}] {plan.note}")
        print("  sources:")
        for source in plan.sources:
            print("   -", source)
        print("  outputs:")
        for output in plan.outputs:
            print("   -", output)


if __name__ == "__main__":
    main()
