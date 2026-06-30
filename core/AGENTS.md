# AGENTS.md — core/ (Model Correctness & Numerical Stability)

This folder defines the scientific model (pool mechanics + agent rules). Changes here can invalidate results.

## Absolute priorities
1) Correctness > speed
2) Explicit assumptions > implicit behavior
3) Numerical stability > clever algebra

## Math conventions (must be consistent)
- Always state whether you use price P or sqrt-price S, and the conversion.
- Always state which token is numéraire in value/PnL metrics.
- Tick boundary handling must be explicit; avoid off-by-one tick ambiguity.

## Before editing any math
- Identify invariants you might violate.
- Add or update a test that would catch that violation.
- Add inline comments for tricky transitions (tick crossing, fee-on-input, rounding rules).

## Rounding & epsilons
- Centralize eps/tolerance constants; do not sprinkle magic eps across files.
- Every epsilon must be justified by scale and floating-point behavior.

## Inputs and config
- Validate config early: types, ranges, required keys.
- Fail loudly with actionable error messages if config is invalid.

## Logging
- If behavior changes, add a diagnostic that helps verify it (summary stats, conserved quantities, etc.).
