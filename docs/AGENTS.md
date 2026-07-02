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
- CLI scenario runs also maintain `abm_results/scenarios/<scenario_name>/latest_run.json`, which points to the newest run folder under `runs/`.
- When naming concrete artifacts, prefer the exact files written by the current runners: `runs/<run_id>/config_snapshot.yml`, `runs/<run_id>/metadata.json`, and `runs/<run_id>/summary.csv`.
- If a plot/table is “canonical”, add the source YAML file to `configs/scenarios/` or `configs/experiments/` and reference it.

## No silent epistemic debt
If you’re unsure about a formula or convention, document the uncertainty explicitly:
- “Working assumption: ...”
- “To verify: ...”
- “Alternative: ...”
