"""Experiment design utilities (sampling plans) for ABM parameter exploration.

This module defines a small, reproducible interface to generate parameter points
for experiments (grid / LHS / Sobol / Saltelli) with mixed continuous/discrete
spaces.

Notes
-----
- These helpers are intentionally "scientific": deterministic given a seed and
  explicit about transforms, bounds, and discrete value mappings.
- This module does not run the simulator; it only generates point lists and
  stable hashes used by scripts that do.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.stats import qmc

ParamKind = Literal["continuous", "discrete"]
TransformKind = Literal["linear", "log"]
CastKind = Literal["int", "float"]
DesignType = Literal["grid", "lhs", "sobol", "sobol_saltelli", "adaptive_refine", "bayesopt"]


def _to_hashable_json(value: Any) -> Any:
    """Convert nested values to deterministic JSON-safe primitives.

    Parameters
    ----------
    value
        Arbitrary nested Python value (may include Path, NumPy scalars, etc.).

    Returns
    -------
    Any
        A JSON-serializable structure composed of dict/list/str/int/float/bool/None.

    Notes
    -----
    - NaN and infinities are mapped to strings ("NaN", "Infinity", "-Infinity") so that
      hashes are stable and JSON encoding is valid with `allow_nan=False`.

    Examples
    --------
    >>> _to_hashable_json({"a": np.float64(1.0), "p": Path("x")})
    {'a': 1.0, 'p': 'x'}
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        if np.isnan(v):
            return "NaN"
        if np.isposinf(v):
            return "Infinity"
        if np.isneginf(v):
            return "-Infinity"
        return v
    if isinstance(value, dict):
        return {str(k): _to_hashable_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_hashable_json(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return value
    return str(value)


def stable_content_hash(payload: Mapping[str, Any], *, n_hex: int = 16) -> str:
    """Compute a stable hash for a JSON-like payload.

    Parameters
    ----------
    payload
        Mapping to hash. Keys are sorted and values canonicalized.
    n_hex
        Number of hex characters to return from the SHA256 digest.

    Returns
    -------
    str
        Short hex digest prefix.

    Notes
    -----
    - Uses UTF-8 JSON with `sort_keys=True` and minimal separators.
    - Intended for cache fingerprints; not a cryptographic guarantee.

    Examples
    --------
    >>> h1 = stable_content_hash({"b": 2, "a": 1})
    >>> h2 = stable_content_hash({"a": 1, "b": 2})
    >>> h1 == h2
    True
    """
    if n_hex <= 0:
        raise ValueError("n_hex must be positive.")
    canon = _to_hashable_json(dict(payload))
    raw = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[: int(n_hex)]


@dataclass(frozen=True)
class ParameterSpec:
    """A single parameter definition in an experiment design space."""

    name: str
    kind: ParamKind
    bounds: Optional[Tuple[float, float]] = None
    values: Optional[Tuple[float | int, ...]] = None
    transform: TransformKind = "linear"
    cast: Optional[CastKind] = None

    def validate(self) -> None:
        """Validate the parameter spec.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - Continuous params must provide `bounds=(lo, hi)` with `hi > lo`.
        - Log transforms require `lo > 0`.
        - Discrete params must provide a non-empty `values` list.

        Examples
        --------
        >>> ParameterSpec(name="k", kind="continuous", bounds=(0.0, 1.0)).validate()
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name must be a non-empty string.")
        if self.kind not in ("continuous", "discrete"):
            raise ValueError(f"Invalid kind for parameter {self.name!r}: {self.kind!r}")

        if self.kind == "continuous":
            if self.bounds is None or len(self.bounds) != 2:
                raise ValueError(f"Continuous parameter {self.name!r} must define bounds: [lo, hi].")
            lo, hi = float(self.bounds[0]), float(self.bounds[1])
            if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                raise ValueError(f"Invalid bounds for {self.name!r}: lo={lo}, hi={hi}.")
            if self.transform == "log" and lo <= 0.0:
                raise ValueError(f"Log-transform requires lo > 0 for {self.name!r} (got lo={lo}).")
            if self.transform not in ("linear", "log"):
                raise ValueError(f"Invalid transform for {self.name!r}: {self.transform!r}")
        else:
            if self.values is None or len(self.values) <= 0:
                raise ValueError(f"Discrete parameter {self.name!r} must define a non-empty values list.")
            if self.cast is not None and self.cast not in ("int", "float"):
                raise ValueError(f"Invalid cast for {self.name!r}: {self.cast!r}")

        # Optional values for continuous are allowed (used by grid designs).
        if self.values is not None and len(self.values) <= 0:
            raise ValueError(f"Parameter {self.name!r} values list cannot be empty.")


@dataclass(frozen=True)
class MetricSpec:
    """Scalar outcomes and histogram settings to compute from simulation outputs."""

    pnl_metrics: Tuple[str, ...]
    pnl_quantiles: Tuple[float, ...]
    include_fee_hist: bool
    fee_hist_bins: int
    include_sr_dex_share_hist: bool
    sr_dex_share_hist_bins: int


@dataclass(frozen=True)
class DesignSpec:
    """Design definition (how to generate points in the space)."""

    type: DesignType
    n_points: Optional[int] = None
    n_base: Optional[int] = None
    seed: Optional[int] = None
    # Sequential designs (optional; not all runners must support them).
    n_init: Optional[int] = None
    batch_size: Optional[int] = None
    target_metric: Optional[str] = None
    direction: Literal["maximize", "minimize"] = "maximize"
    regime_threshold: Optional[float] = None


@dataclass(frozen=True)
class ExperimentSpec:
    """Parsed experiment YAML specification."""

    version: int
    name: str
    base_config: Path
    outputs_root: Path
    outputs_tag: str
    keep_worker_tmp: bool
    seed_base: int
    common_seeds: bool
    runs_per_point: int
    max_workers: int
    light_mode: bool
    visualize: bool
    verbose: bool
    metrics: MetricSpec
    space: Tuple[ParameterSpec, ...]
    design: DesignSpec


@dataclass(frozen=True)
class DesignPoint:
    """A single parameter point produced by a design generator."""

    point_id: int
    values: Mapping[str, float | int]
    indices: Mapping[str, int]
    meta: Mapping[str, Any]


def load_experiment_spec(path: Path) -> ExperimentSpec:
    """Load and validate an experiment YAML file.

    Parameters
    ----------
    path
        Path to the experiment YAML file.

    Returns
    -------
    ExperimentSpec
        Parsed, validated experiment specification.

    Notes
    -----
    - This function is strict by design: invalid or ambiguous experiment configs
      should fail loudly before any expensive simulations start.
    - Paths are resolved relative to the experiment file location.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> p = Path(tempfile.gettempdir()) / "experiment.yml"
    >>> _ = p.write_text(\"\"\"version: 1\\nname: demo\\nbase_config: configs/scenarios/section4_microstructure_model0_static.yml\\noutputs: {root: abm_results/experiments_runs}\\nseed: {seed_base: 1, runs_per_point: 2}\\ncompute: {max_workers: 1}\\nmetrics: {pnl_metrics: [lp_pnl_passive], pnl_quantiles: [0.5], include_fee_hist: false, fee_hist_bins: 10, include_sr_dex_share_hist: false, sr_dex_share_hist_bins: 10}\\ndesign: {type: lhs, n_points: 4}\\nspace: [{name: k_sigma, kind: continuous, bounds: [0.0, 1.0]}]\\n\"\"\", encoding=\"utf-8\")  # doctest: +SKIP
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must parse to a mapping (YAML dict).")

    version = int(raw.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported experiment version {version}. Expected version: 1.")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("Experiment YAML must define a non-empty 'name'.")

    base_config_raw = raw.get("base_config")
    if not isinstance(base_config_raw, str) or not base_config_raw:
        raise ValueError("Experiment YAML must define 'base_config' as a path string.")
    base_config = (path.parent / Path(base_config_raw)).expanduser().resolve()

    outputs = raw.get("outputs") or {}
    if not isinstance(outputs, dict):
        raise ValueError("'outputs' must be a mapping.")
    outputs_root_raw = outputs.get("root", "abm_results/experiments_runs")
    outputs_root = (path.parent / Path(str(outputs_root_raw))).expanduser().resolve()
    outputs_tag = str(outputs.get("tag") or name).strip()
    if not outputs_tag:
        outputs_tag = name
    keep_worker_tmp = bool(outputs.get("keep_worker_tmp", False))

    seed = raw.get("seed") or {}
    if not isinstance(seed, dict):
        raise ValueError("'seed' must be a mapping.")
    seed_base = int(seed.get("seed_base", 1))
    common_seeds = bool(seed.get("common_seeds", False))
    runs_per_point = int(seed.get("runs_per_point", 15))
    if runs_per_point <= 0:
        raise ValueError("'seed.runs_per_point' must be positive.")

    compute = raw.get("compute") or {}
    if not isinstance(compute, dict):
        raise ValueError("'compute' must be a mapping.")
    max_workers = int(compute.get("max_workers", 1))
    if max_workers <= 0:
        raise ValueError("'compute.max_workers' must be positive.")
    light_mode = bool(compute.get("light_mode", True))
    visualize = bool(compute.get("visualize", False))
    verbose = bool(compute.get("verbose", False))

    metrics_raw = raw.get("metrics") or {}
    if not isinstance(metrics_raw, dict):
        raise ValueError("'metrics' must be a mapping.")
    pnl_metrics_raw = metrics_raw.get("pnl_metrics") or [
        "lp_pnl_passive",
        "lp_pnl_active",
        "arb_pnl_cum",
        "noise_trader_pnl_cum",
    ]
    if not isinstance(pnl_metrics_raw, list) or not pnl_metrics_raw:
        raise ValueError("'metrics.pnl_metrics' must be a non-empty list.")
    pnl_metrics = tuple(str(v) for v in pnl_metrics_raw)

    pnl_quantiles_raw = metrics_raw.get("pnl_quantiles") or [0.10, 0.50, 0.90]
    if not isinstance(pnl_quantiles_raw, list) or not pnl_quantiles_raw:
        raise ValueError("'metrics.pnl_quantiles' must be a non-empty list.")
    pnl_quantiles = tuple(float(v) for v in pnl_quantiles_raw)
    for q in pnl_quantiles:
        if not (0.0 <= float(q) <= 1.0):
            raise ValueError(f"Invalid quantile in metrics.pnl_quantiles: {q}")

    include_fee_hist = bool(metrics_raw.get("include_fee_hist", True))
    fee_hist_bins = int(metrics_raw.get("fee_hist_bins", 60))
    if fee_hist_bins <= 1:
        raise ValueError("'metrics.fee_hist_bins' must be > 1.")
    include_sr = bool(metrics_raw.get("include_sr_dex_share_hist", True))
    sr_hist_bins = int(metrics_raw.get("sr_dex_share_hist_bins", fee_hist_bins))
    if sr_hist_bins <= 1:
        raise ValueError("'metrics.sr_dex_share_hist_bins' must be > 1.")

    metrics = MetricSpec(
        pnl_metrics=pnl_metrics,
        pnl_quantiles=pnl_quantiles,
        include_fee_hist=include_fee_hist,
        fee_hist_bins=fee_hist_bins,
        include_sr_dex_share_hist=include_sr,
        sr_dex_share_hist_bins=sr_hist_bins,
    )

    design_raw = raw.get("design") or {}
    if not isinstance(design_raw, dict):
        raise ValueError("'design' must be a mapping.")
    design_type = str(design_raw.get("type", "")).strip()
    if design_type not in (
        "grid",
        "lhs",
        "sobol",
        "sobol_saltelli",
        "adaptive_refine",
        "bayesopt",
    ):
        raise ValueError(f"Unsupported design.type: {design_type!r}")
    n_points = design_raw.get("n_points")
    n_base = design_raw.get("n_base")
    design_seed = design_raw.get("seed")
    n_init = design_raw.get("n_init")
    batch_size = design_raw.get("batch_size")
    target_metric = design_raw.get("target_metric")
    direction = str(design_raw.get("direction", "maximize")).strip().lower()
    if direction not in ("maximize", "minimize"):
        raise ValueError("'design.direction' must be 'maximize' or 'minimize'.")
    regime_threshold = design_raw.get("regime_threshold")

    design = DesignSpec(
        type=design_type,  # type: ignore[arg-type]
        n_points=None if n_points is None else int(n_points),
        n_base=None if n_base is None else int(n_base),
        seed=None if design_seed is None else int(design_seed),
        n_init=None if n_init is None else int(n_init),
        batch_size=None if batch_size is None else int(batch_size),
        target_metric=None if target_metric is None else str(target_metric),
        direction=direction,  # type: ignore[arg-type]
        regime_threshold=None if regime_threshold is None else float(regime_threshold),
    )

    if design.type in ("lhs", "sobol", "adaptive_refine", "bayesopt") and (design.n_points is None or design.n_points <= 0):
        raise ValueError(f"design.type={design.type!r} requires a positive design.n_points.")
    if design.type == "sobol_saltelli" and (design.n_base is None or design.n_base <= 0):
        # Allow n_points as an alias for n_base for convenience.
        if design.n_points is not None and design.n_points > 0:
            design = DesignSpec(
                type=design.type,
                n_points=design.n_points,
                n_base=int(design.n_points),
                seed=design.seed,
                n_init=design.n_init,
                batch_size=design.batch_size,
                target_metric=design.target_metric,
                direction=design.direction,
                regime_threshold=design.regime_threshold,
            )
        else:
            raise ValueError("design.type='sobol_saltelli' requires design.n_base (or design.n_points as an alias).")
    if design.type in ("adaptive_refine", "bayesopt") and not design.target_metric:
        raise ValueError(f"design.type={design.type!r} requires a non-empty design.target_metric.")

    space_raw = raw.get("space")
    if not isinstance(space_raw, list) or not space_raw:
        raise ValueError("'space' must be a non-empty list.")
    specs: List[ParameterSpec] = []
    seen: set[str] = set()
    for entry in space_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each entry in 'space' must be a mapping.")
        pname = str(entry.get("name", "")).strip()
        if not pname:
            raise ValueError("Each space entry must define a non-empty 'name'.")
        if pname in seen:
            raise ValueError(f"Duplicate parameter name in space: {pname!r}")
        seen.add(pname)
        pkind = str(entry.get("kind", "")).strip()
        if pkind not in ("continuous", "discrete"):
            raise ValueError(f"Invalid kind for parameter {pname!r}: {pkind!r}")
        bounds = None
        if entry.get("bounds") is not None:
            b = entry.get("bounds")
            if not isinstance(b, (list, tuple)) or len(b) != 2:
                raise ValueError(f"Parameter {pname!r} bounds must be a 2-element list.")
            bounds = (float(b[0]), float(b[1]))
        values = None
        if entry.get("values") is not None:
            v = entry.get("values")
            if not isinstance(v, list) or not v:
                raise ValueError(f"Parameter {pname!r} values must be a non-empty list.")
            values = tuple(v)
        transform = str(entry.get("transform", "linear")).strip().lower()
        cast = entry.get("cast")
        cast_out: Optional[CastKind] = None
        if cast is not None:
            cast_out = str(cast).strip().lower()  # type: ignore[assignment]
        spec = ParameterSpec(
            name=pname,
            kind=pkind,  # type: ignore[arg-type]
            bounds=bounds,
            values=values,
            transform=transform,  # type: ignore[arg-type]
            cast=cast_out,
        )
        spec.validate()
        specs.append(spec)

    return ExperimentSpec(
        version=version,
        name=name,
        base_config=base_config,
        outputs_root=outputs_root,
        outputs_tag=outputs_tag,
        keep_worker_tmp=keep_worker_tmp,
        seed_base=seed_base,
        common_seeds=common_seeds,
        runs_per_point=runs_per_point,
        max_workers=max_workers,
        light_mode=light_mode,
        visualize=visualize,
        verbose=verbose,
        metrics=metrics,
        space=tuple(specs),
        design=design,
    )


def map_unit_to_point(space: Sequence[ParameterSpec], u: Sequence[float]) -> Tuple[Dict[str, float | int], Dict[str, int]]:
    """Map a unit-hypercube sample to parameter values.

    Parameters
    ----------
    space
        Parameter specifications in deterministic order.
    u
        Unit cube coordinates in [0, 1), one per parameter (same order as `space`).

    Returns
    -------
    (values, indices)
        `values` is a mapping `param -> numeric value`.
        `indices` stores `param -> index` for discrete parameters (index into `values` list).

    Notes
    -----
    - Continuous parameters use linear or log mapping from bounds.
    - Discrete parameters map `u` to an integer index by equal-probability binning.
    - Returned values are plain Python scalars (int/float), suitable for JSON/CSV.

    Examples
    --------
    >>> space = [ParameterSpec(name="k", kind="continuous", bounds=(0.0, 1.0))]
    >>> v, idx = map_unit_to_point(space, [0.25])
    >>> 0.0 <= float(v["k"]) <= 1.0
    True
    """
    if len(u) != len(space):
        raise ValueError(f"Expected u of length {len(space)} but got {len(u)}.")
    values: Dict[str, float | int] = {}
    indices: Dict[str, int] = {}
    for spec, ui in zip(space, u):
        ui_f = float(ui)
        ui_f = min(max(ui_f, 0.0), np.nextafter(1.0, 0.0))
        if spec.kind == "continuous":
            if spec.bounds is None:
                raise ValueError(f"Continuous parameter {spec.name!r} missing bounds.")
            lo, hi = float(spec.bounds[0]), float(spec.bounds[1])
            if spec.transform == "log":
                x = math.exp(math.log(lo) + ui_f * (math.log(hi) - math.log(lo)))
            else:
                x = lo + ui_f * (hi - lo)
            values[spec.name] = float(x)
        else:
            if spec.values is None:
                raise ValueError(f"Discrete parameter {spec.name!r} missing values.")
            k = int(len(spec.values))
            j = int(min(k - 1, math.floor(ui_f * k)))
            raw_val = spec.values[j]
            if spec.cast == "int":
                values[spec.name] = int(round(float(raw_val)))
            else:
                # Default: preserve ints if the provided value is an int-like.
                if isinstance(raw_val, (int, np.integer)):
                    values[spec.name] = int(raw_val)
                else:
                    values[spec.name] = float(raw_val)
            indices[spec.name] = int(j)
    return values, indices


def generate_design_points(
    *,
    space: Sequence[ParameterSpec],
    design: DesignSpec,
    seed_base: int,
) -> Tuple[List[DesignPoint], Dict[str, Any]]:
    """Generate design points for an experiment.

    Parameters
    ----------
    space
        Parameter specifications.
    design
        Design spec defining the generator type and budgets.
    seed_base
        Base seed used when `design.seed` is not provided.

    Returns
    -------
    (points, meta)
        `points` is a deterministic list of DesignPoint objects.
        `meta` is a JSON-serializable mapping describing the design instance
        (useful for downstream analysis, e.g., Saltelli indexing).

    Notes
    -----
    - For `grid`, values must be explicitly provided for every parameter (either
      via discrete `values` or continuous `values` lists).
    - For `sobol_saltelli`, we generate A/B/AB_i sets in a fixed order:
        A (N), B (N), then AB_0..AB_{d-1} (each N).

    Examples
    --------
    >>> space = [ParameterSpec(name="k", kind="continuous", bounds=(0.0, 1.0))]
    >>> design = DesignSpec(type="lhs", n_points=4, seed=1)
    >>> pts, meta = generate_design_points(space=space, design=design, seed_base=1)
    >>> len(pts) == 4
    True
    """
    for spec in space:
        spec.validate()
    d = int(len(space))
    if d <= 0:
        raise ValueError("Space must contain at least one parameter.")

    seed = int(seed_base if design.seed is None else design.seed)
    meta: Dict[str, Any] = {"type": str(design.type), "seed": int(seed)}

    points: List[DesignPoint] = []

    if design.type == "grid":
        value_lists: List[Sequence[float | int]] = []
        for spec in space:
            if spec.values is None:
                raise ValueError(
                    f"Grid design requires explicit 'values' for parameter {spec.name!r} "
                    "(continuous bounds alone are ambiguous)."
                )
            value_lists.append(list(spec.values))
        sizes = [len(v) for v in value_lists]
        if any(n <= 0 for n in sizes):
            raise ValueError("Grid design has an empty values list for at least one parameter.")
        meta["param_order"] = [spec.name for spec in space]
        meta["sizes"] = sizes

        point_id = 0
        # Deterministic cartesian product in the parameter order listed in YAML.
        # Use streaming iteration to avoid allocating huge intermediate arrays.
        for combo in product(*[range(int(n)) for n in sizes]):
            indices: Dict[str, int] = {}
            values: Dict[str, float | int] = {}
            for spec, j in zip(space, combo):
                j_int = int(j)
                raw_val = spec.values[j_int] if spec.values is not None else None
                if raw_val is None:
                    raise RuntimeError("Internal error: missing values in grid mapping.")
                if spec.kind == "discrete":
                    indices[spec.name] = j_int
                    if spec.cast == "int":
                        values[spec.name] = int(round(float(raw_val)))
                    else:
                        values[spec.name] = int(raw_val) if isinstance(raw_val, (int, np.integer)) else float(raw_val)
                else:
                    # For continuous grids, keep float.
                    values[spec.name] = float(raw_val)
            points.append(DesignPoint(point_id=point_id, values=values, indices=indices, meta={}))
            point_id += 1
        meta["n_points"] = int(point_id)
        return points, meta

    if design.type == "lhs":
        if design.n_points is None or design.n_points <= 0:
            raise ValueError("LHS design requires design.n_points > 0.")
        engine = qmc.LatinHypercube(d=d, seed=int(seed))
        u = engine.random(n=int(design.n_points))
        meta["n_points"] = int(design.n_points)
    elif design.type == "sobol":
        if design.n_points is None or design.n_points <= 0:
            raise ValueError("Sobol design requires design.n_points > 0.")
        engine = qmc.Sobol(d=d, scramble=True, seed=int(seed))
        u = engine.random(n=int(design.n_points))
        meta["n_points"] = int(design.n_points)
        if (design.n_points & (design.n_points - 1)) != 0:
            meta["note"] = "Sobol performs best when n_points is a power of two."
    elif design.type == "sobol_saltelli":
        n_base = design.n_base if design.n_base is not None else design.n_points
        if n_base is None or n_base <= 0:
            raise ValueError("Saltelli design requires design.n_base > 0.")
        engine = qmc.Sobol(d=2 * d, scramble=True, seed=int(seed))
        u2 = engine.random(n=int(n_base))
        uA = u2[:, :d]
        uB = u2[:, d:]
        meta["n_base"] = int(n_base)
        meta["n_points"] = int(n_base) * (2 + d)
        meta["saltelli"] = {"n_base": int(n_base), "d": int(d), "order": "A,B,AB_i"}

        def _map_rows(mat: np.ndarray, *, role: str, i_param: Optional[int]) -> None:
            nonlocal points
            start_id = len(points)
            for r in range(mat.shape[0]):
                vals, idxs = map_unit_to_point(space, mat[r, :].tolist())
                meta_row = {"role": role}
                if i_param is not None:
                    meta_row["i_param"] = int(i_param)
                points.append(
                    DesignPoint(
                        point_id=start_id + int(r),
                        values=vals,
                        indices=idxs,
                        meta=meta_row,
                    )
                )

        _map_rows(uA, role="A", i_param=None)
        _map_rows(uB, role="B", i_param=None)
        for i in range(d):
            uAB = np.array(uA, copy=True)
            uAB[:, i] = uB[:, i]
            _map_rows(uAB, role="AB", i_param=i)
        return points, meta
    else:
        raise ValueError(f"Design type {design.type!r} is not supported by the static generator.")

    for i in range(int(u.shape[0])):
        vals, idxs = map_unit_to_point(space, u[i, :].tolist())
        points.append(DesignPoint(point_id=int(i), values=vals, indices=idxs, meta={}))
    return points, meta


def experiment_yaml_content_hash(path: Path) -> str:
    """Hash the *content* of an experiment YAML after parsing.

    Parameters
    ----------
    path
        Experiment YAML path.

    Returns
    -------
    str
        Short stable hash of the parsed YAML mapping (order-insensitive for keys).

    Notes
    -----
    - This is intended for cache fingerprinting; it ignores YAML formatting details.

    Examples
    --------
    >>> isinstance(experiment_yaml_content_hash(Path("configs/scenarios/section4_microstructure_model0_static.yml")), str)  # doctest: +SKIP
    True
    """
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must parse to a mapping for hashing.")
    return stable_content_hash(raw, n_hex=16)
