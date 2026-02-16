# AGENTS.md — ABM_Uni_v3 (Scientific Verification & Exploration)

This repository is a scientific instrument: changes must make the simulator **more correct, more reproducible, and more falsifiable**. Prefer clarity over cleverness, and tests/diagnostics over vibes.

## Mission
- Primary goal: enable **scientific verification** (replication, invariants, sanity checks) and **exploration** (parameter sweeps, counterfactuals).
- Default posture: be skeptical. When changing behavior, also add checks that would catch if the change is wrong.

## Repo map (quick orientation)
- `scripts/run.py` is the main simulation runner; simulations are configured via YAML scenario files. :contentReference[oaicite:1]{index=1}
- Example scenarios live under `abm_results/scenarios/` (e.g. `abm_results/scenarios/test.yml`). :contentReference[oaicite:2]{index=2}
- Per-scenario outputs (plots/logs/results) are written under `abm_results/scenarios/<scenario_name>/`. :contentReference[oaicite:3]{index=3}
- Core mechanics:
  - v3 pool math: `core/uniswapv3_pool.py`
  - LP agents and position accounting: `core/agents.py`
  - reference market + utilities: `core/utils.py`

## Non-negotiables (scientific hygiene)
1. **Reproducibility first**
   - All experiments must be reproducible from a scenario YAML and a seed.
   - If you add RNG usage, thread the seed explicitly and keep it discoverable in logs/outputs.
   - Never rely on wall-clock time or non-deterministic iteration order for scientific results.

2. **Every behavioral change needs a “why it’s true” artifact**
   - Add or update at least one of:
     - a unit test / property test,
     - an invariant check (assertion with informative message),
     - a diagnostic plot/statistic written to results,
     - or a documented derivation/assumption in `docs/`.

3. **Do not silently overwrite results**
   - Treat result folders as experimental records.
   - If generating new outputs, use a new `config_name` (scenario name) or a timestamped/numbered subfolder.

4. **Keep the simulator falsifiable**
   - Prefer exposing parameters in YAML over hard-coding.
   - Prefer explicit assumptions to “magic numbers”.

## How to run (local workflow)
- Activate conda env: `conda activate main`
- Run minimal scenario (example): `python run.py --config abm_results/scenarios/test.yml` :contentReference[oaicite:5]{index=5}
- Run tests: `pytest -q` :contentReference[oaicite:6]{index=6}

If you add new CLI flags or scenario keys, update `docs/` and ensure `test.yml` stays runnable.

## What to do when asked to change something
Always follow this sequence:
1. **Restate the scientific claim** in one sentence (what should become measurably true?).
2. Identify the *minimum* mechanism(s) that must change.
3. Add/extend a check that would fail before the change and pass after.
4. Implement.
5. Re-run the smallest scenario + tests.
6. Summarize: what changed, which evidence supports correctness, what assumptions were introduced.

## Coding standards (pragmatic, scientific)
- Keep functions small and explicit; prefer readability over micro-optimizations.
- Add docstrings for nontrivial math (state units! token0/token1, sqrt-price S vs price P).
- Avoid “silent” behavior: validate inputs early (especially YAML configs).
- When performance matters, measure first; if optimizing, keep a clear baseline and add regression checks.

## Verification checklist (use these as guardrails)
When touching pool math / swaps / fee allocation:
- Check invariants: no negative active liquidity beyond eps; monotonic tick movement inside segment; fee-on-input consistency.
- Add tests around boundary cases (near tick boundaries, zero liquidity “desert”, tiny swaps, large swaps).

When touching LP accounting (PnL, IL/LVR, fees):
- Verify units and numéraire (token1 valuation).
- Add at least one conservation-style check (e.g., value changes match realized flows + fees within tolerance).

When touching stochastic components:
- Ensure seeds control all RNGs used (Python `random` + NumPy).
- Add a distribution sanity test (mean/variance within tolerance over many draws) only if it’s stable enough.

## Results & diagnostics conventions
- Outputs should be written under the scenario’s results root (`abm_results/scenarios/<scenario_name>/...`). :contentReference[oaicite:7]{index=7}
- Any new metric should be:
  - logged in a machine-readable form (CSV/JSON),
  - and optionally plotted (Plotly is already used in `run.py`).

## “Exploration mode” patterns (approved)
- Parameter sweeps should live in scripts like `run_multiple.py` / dashboard runners, but always:
  - record the full config used,
  - record the code version (git commit hash if available),
  - and output a single summary table to compare runs.

## What NOT to do
- Don’t refactor large sections without preserving outputs/behavior or adding equivalence checks.
- Don’t change defaults in ways that break existing scenarios without a migration note.
- Don’t remove diagnostics just to make code “cleaner” unless replaced by better ones.

## Communication format for agent output (when making a PR / patch)
Provide:
- **Claim:** what should now be true
- **Evidence:** tests run + key metrics/plots (paths)
- **Risk:** what might still be wrong / assumptions
- **Reproduce:** exact command(s) + scenario file used
