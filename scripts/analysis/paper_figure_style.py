"""Shared style-configuration helpers for paper figure generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def merge_style(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""

    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_style(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _as_mapping(value: Any, *, label: str, path: Path) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {label} in style config {path}")
    return value


def load_figure_style(
    path: Path,
    default_style: Mapping[str, Any] | None = None,
    *,
    figure_key: str | None = None,
) -> Dict[str, Any]:
    """Load global paper style plus optional per-figure overrides.

    The YAML schema is intentionally backward compatible with the earlier
    ``paper_figure_style`` file used by the block-size Plotly generators:

    - ``paper_figure_style`` holds global defaults shared by all figures.
    - ``figures.<figure_key>`` holds overrides for one paper figure.

    If ``paper_figure_style`` is absent, top-level keys other than ``figures``
    are treated as the global style, preserving older flat config files.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing paper style config: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping at top level of style config: {path}")

    if "paper_figure_style" in data:
        global_style = _as_mapping(data.get("paper_figure_style"), label="paper_figure_style", path=path)
    else:
        global_style = {key: value for key, value in data.items() if key != "figures"}

    style: Dict[str, Any] = dict(default_style or {})
    style = merge_style(style, global_style)

    if figure_key is not None:
        figures = _as_mapping(data.get("figures", {}), label="figures", path=path)
        override = _as_mapping(figures.get(figure_key, {}), label=f"figures.{figure_key}", path=path)
        style = merge_style(style, override)

    return style
