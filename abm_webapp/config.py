"""
Strict configuration validation for ABM webapp scenarios.

Treats YAML as untrusted input: safe loader, size limits, type/bounds checks,
rejection of unknown fields.  Resource-heavy settings are capped with sensible
defaults that can be raised via environment variables.

All public helpers return ``(validated_dict, error_message)`` tuples so callers
can surface clear diagnostics to the UI.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# Size / resource guards
# ---------------------------------------------------------------------------
MAX_YAML_BYTES: int = 512_000  # 512 KB – more than enough for any scenario
# Keep the default cap above the repository's bundled yearly scenarios so the
# webapp can run first-party configs out of the box. Users can still lower or
# raise the cap explicitly via the environment.
MAX_T: int = int(os.environ.get("ABM_MAX_T", 1_000_000))
MAX_N_LP: int = int(os.environ.get("ABM_MAX_N_LP", 2_000))
MAX_BLOCK_TIME: int = int(os.environ.get("ABM_MAX_BLOCK_TIME", 120))

# ---------------------------------------------------------------------------
# Known top-level keys and their types (shallow)
# ---------------------------------------------------------------------------
_TOP_LEVEL_KEYS: Set[str] = {"fee_mode", "simulate", "scenario"}

# ---------------------------------------------------------------------------
# Critical simulate() parameters – type + optional bounds
# Each entry: (python_type, min, max, required)
# None for min/max means "no bound".
# ---------------------------------------------------------------------------
_SIMULATE_BOUNDS: Dict[str, Tuple[type, Optional[float], Optional[float], bool]] = {
    "T": (int, 1, MAX_T, True),
    "seed": (int, None, None, True),
    "block_time": (int, 2, MAX_BLOCK_TIME, True),
    "N_LP": (int, 0, MAX_N_LP, False),
    "passive_lp_share": (float, 0.0, 1.0, False),
    "cex_sigma": (float, 0.0, None, True),
    "cex_mu": (float, None, None, True),
    "f0": (float, 0.0, 1.0, False),
    "f_min": (float, 0.0, 1.0, False),
    "f_max": (float, 0.0, 1.0, False),
    "fee_use_ewma": (bool, None, None, False),
    "asymmetric_fee_slope": (float, 0.0, None, False),
    "smart_trades_per_second": (float, 0.0, None, False),
    "noise_trades_per_second": (float, 0.0, None, False),
    "narrow_mints_per_second": (float, 0.0, None, False),
    "passive_mints_per_second": (float, 0.0, None, False),
    "passive_burns_per_second": (float, 0.0, None, False),
    "smart_trades_per_block": (float, 0.0, None, False),
    "noise_trades_per_block": (float, 0.0, None, False),
    "tau_seconds": (float, 0.0, None, False),
    "fee_mode": (str, None, None, False),
    "p_jit": (float, 0.0, 1.0, False),
    "N_jit": (int, 0, None, False),
    "liquidity_perc_jit": (float, 0.0, 1.0, False),
    "slippage_tolerance": (float, 0.0, 1.0, False),
    "cex_heston_kappa": (float, 0.0, None, False),
    "cex_heston_theta": (float, 0.0, None, False),
    "cex_heston_sigma_v": (float, 0.0, None, False),
    "cex_heston_rho": (float, -1.0, 1.0, False),
    "cex_heston_v0": (float, 0.0, None, False),
}

_VALID_FEE_MODES: Set[str] = {
    "static",
    "volatility_cex",
    "volatility_dex",
    "toxicity",
    "lvr_fee_ewma",
    "linear_asymmetric",
}
_VALID_SIGMA_MODES: Set[str] = {"static", "heston"}


def safe_load_yaml(text: str, *, max_bytes: int = MAX_YAML_BYTES) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse YAML from a string with size guard and safe loader.

    Returns
    -------
    (parsed_dict | None, error_message)
    """
    if not isinstance(text, str):
        return None, "Config must be a string."
    if len(text.encode("utf-8", errors="replace")) > max_bytes:
        return None, f"YAML input exceeds {max_bytes:,} bytes limit."
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, f"YAML parse error: {exc}"
    if data is None:
        return None, "YAML is empty."
    if not isinstance(data, dict):
        return None, "YAML root must be a mapping."
    return data, ""


def validate_scenario(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Full validation pipeline for a scenario YAML string.

    Returns
    -------
    (canonical_dict | None, error_message)
        On success the canonical dict is the cleaned, validated scenario.
    """
    data, err = safe_load_yaml(text)
    if err:
        return None, err

    assert data is not None  # mypy

    # ── Reject unknown top-level keys ──
    unknown_top = sorted(set(data.keys()) - _TOP_LEVEL_KEYS)
    if unknown_top:
        return None, f"Unknown top-level keys: {unknown_top}. Allowed: {sorted(_TOP_LEVEL_KEYS)}"

    # ── fee_mode at top level propagation ──
    top_fee_mode = data.get("fee_mode")
    simulate_block = data.get("simulate")
    if simulate_block is None:
        return None, "Missing required 'simulate' section."
    if not isinstance(simulate_block, dict):
        return None, "'simulate' must be a mapping."

    # Propagate top-level fee_mode into simulate block if not conflicting
    if top_fee_mode is not None:
        inner_fee_mode = simulate_block.get("fee_mode")
        if inner_fee_mode is not None and str(inner_fee_mode) != str(top_fee_mode):
            return None, (
                f"Conflicting fee_mode: top-level='{top_fee_mode}' vs "
                f"simulate.fee_mode='{inner_fee_mode}'."
            )
        simulate_block = dict(simulate_block, fee_mode=top_fee_mode)

    # ── Validate fee_mode value ──
    fee_mode_val = simulate_block.get("fee_mode")
    if fee_mode_val is not None and str(fee_mode_val) not in _VALID_FEE_MODES:
        return None, f"Invalid fee_mode '{fee_mode_val}'. Expected one of {sorted(_VALID_FEE_MODES)}."

    sigma_mode_val = simulate_block.get("cex_sigma_mode")
    if sigma_mode_val is not None and str(sigma_mode_val) not in _VALID_SIGMA_MODES:
        return None, (
            f"Invalid cex_sigma_mode '{sigma_mode_val}'. "
            f"Expected one of {sorted(_VALID_SIGMA_MODES)}."
        )

    # ── Type + bounds checks on critical parameters ──
    errors: List[str] = []
    for key, (expected_type, lo, hi, required) in _SIMULATE_BOUNDS.items():
        val = simulate_block.get(key)
        if val is None:
            if required:
                errors.append(f"Missing required parameter 'simulate.{key}'.")
            continue
        # Type coercion / check
        try:
            if expected_type is int:
                coerced = int(val)
            elif expected_type is float:
                coerced = float(val)
            elif expected_type is str:
                coerced = str(val)
            elif expected_type is bool:
                if not isinstance(val, bool):
                    raise ValueError
                coerced = val
            else:
                coerced = val
        except (TypeError, ValueError):
            errors.append(f"simulate.{key}: expected {expected_type.__name__}, got {type(val).__name__}={val!r}.")
            continue
        if lo is not None and isinstance(coerced, (int, float)) and coerced < lo:
            errors.append(f"simulate.{key}={coerced} is below minimum {lo}.")
        if hi is not None and isinstance(coerced, (int, float)) and coerced > hi:
            errors.append(f"simulate.{key}={coerced} exceeds maximum {hi}.")

    # Additional relational checks that catch common configuration mistakes
    # before the worker process starts.
    f_min = simulate_block.get("f_min")
    f0 = simulate_block.get("f0")
    f_max = simulate_block.get("f_max")
    try:
        f_min_f = float(f_min) if f_min is not None else None
        f0_f = float(f0) if f0 is not None else None
        f_max_f = float(f_max) if f_max is not None else None
    except (TypeError, ValueError):
        f_min_f = None
        f0_f = None
        f_max_f = None
    if f_min_f is not None and f_max_f is not None and f_min_f > f_max_f:
        errors.append(f"simulate.f_min={f_min_f} cannot exceed simulate.f_max={f_max_f}.")
    if (
        f0_f is not None
        and f_min_f is not None
        and f_max_f is not None
        and not (f_min_f <= f0_f <= f_max_f)
    ):
        errors.append(
            f"simulate.f0={f0_f} must lie within [simulate.f_min={f_min_f}, simulate.f_max={f_max_f}]."
        )

    if str(simulate_block.get("cex_sigma_mode", "static")).lower() == "heston":
        required_heston = [
            "cex_heston_kappa",
            "cex_heston_theta",
            "cex_heston_sigma_v",
            "cex_heston_rho",
        ]
        missing_heston = [key for key in required_heston if simulate_block.get(key) is None]
        if missing_heston:
            errors.append(
                "simulate.cex_sigma_mode='heston' requires: "
                + ", ".join(f"simulate.{key}" for key in missing_heston)
                + "."
            )

    if errors:
        return None, "Config validation errors:\n  • " + "\n  • ".join(errors)

    # ── Build canonical output ──
    canonical: Dict[str, Any] = {}
    if top_fee_mode is not None:
        canonical["fee_mode"] = str(top_fee_mode)
    canonical["simulate"] = dict(simulate_block)
    if "scenario" in data:
        canonical["scenario"] = data["scenario"]

    return canonical, ""


def validate_scenario_text(text: str) -> Tuple[bool, str]:
    """
    Thin wrapper returning ``(ok, error)`` for backwards compatibility.
    """
    result, err = validate_scenario(text)
    if err:
        return False, err
    return True, ""


def canonical_yaml(config: Dict[str, Any]) -> str:
    """Dump a validated config dict as deterministic YAML."""
    return yaml.dump(config, default_flow_style=False, sort_keys=True, allow_unicode=True)
