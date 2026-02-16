# AGENTS.md — docs/ (Assumptions, Derivations, Reproduction)

Docs here are not marketing; they are the *lab notebook*.

## What docs must contain
- **Assumptions:** what is assumed (market model, agent behavior, fee model, etc.) and why.
- **Definitions / units:** token0/token1, price vs sqrt-price, tick conventions, numéraire.
- **Derivations / references:** brief derivations for formulas or cite canonical sources.
- **Reproduction recipe:** exact commands + scenario YAML used to reproduce key results.

## Style
- Prefer short, checkable statements over long prose.
- Every formula should define symbols immediately (units too).
- If you state an empirical claim (“X increases with Y”), include the scenario and output path that demonstrates it.

## Change discipline
- Any code change that alters a scientific assumption requires a doc update:
  - add a changelog note in the relevant doc,
  - update the derivation/assumption section,
  - or add a “Known limitations” bullet.

## Output referencing
- Refer to outputs by relative paths under `abm_results/scenarios/<scenario_name>/...`.
- If a plot/table is “canonical”, add the scenario YAML file to `docs/experiments/` (or similar) and reference it.

## No silent epistemic debt
If you’re unsure about a formula or convention, document the uncertainty explicitly:
- “Working assumption: ...”
- “To verify: ...”
- “Alternative: ...”
