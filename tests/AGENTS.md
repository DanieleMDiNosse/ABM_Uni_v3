# AGENTS.md — tests/ (Verification Harness)

This directory is the repo’s truth serum. The purpose of tests here is to detect scientific wrongness early:
math regressions, conservation/invariant violations, seed nondeterminism, and unit mistakes.

## Principles
- Prefer tests that check **invariants / properties** over brittle exact numbers.
- When exact numbers are required, justify them (derivation or stable fixture) and keep tolerances explicit.
- Every bug fix should come with a regression test that fails before the fix.

## What to test (priority order)
1) Determinism & reproducibility
- Same scenario + same seed => identical key outputs within tolerance.
- All RNG sources must be controlled (NumPy + Python random if used).

2) Invariants (pool mechanics)
- No negative liquidity beyond a small epsilon.
- Tick / sqrt-price monotonicity within a swap segment.
- Conservation-style checks: token balances change consistently with swap direction and fees (within tolerance).

3) Boundary cases
- Very small swaps (underflow-ish).
- Very large swaps (crossing multiple ticks).
- Near tick boundary swaps (precision stress).
- Zero-liquidity regions / “deserts”.
- Edge parameters (fee=0, extreme vol, etc.) if allowed by config.

4) Accounting
- LP position value, fees, and PnL: units consistent (token0/token1, price numéraire).
- If you compute IL/LVR or similar metrics: sanity checks (e.g., zero activity => near-zero fees, etc.).

## Tolerances
- Use absolute and relative tolerances intentionally.
- Prefer `math.isclose` / `numpy.testing.assert_allclose` with explicit `rtol`, `atol`.
- Document the scale: "rtol=1e-9 because values are O(1)" vs "atol=1e-8 because small numbers".

## Test style
- Keep tests small. Avoid “do-everything” integration tests unless they are the only way.
- If a test uses scenario YAML, store a minimal YAML fixture and keep it stable.
- Name tests by behavior: `test_swap_conserves_value_with_fees()` not `test_case_12()`.

## Running
Default: `pytest -q`
If adding slow tests, mark them `@pytest.mark.slow` and ensure fast suite remains fast.
