# AGENTS.md — scripts/ (Exploration Without Losing Reproducibility)

Scripts are allowed to be messy *only* if they produce clean, reproducible artifacts.

## Requirements
- Every run writes:
  - scenario config used (full YAML snapshot),
  - seed(s),
  - code version (git commit hash if available),
  - and a single summary table (CSV/Parquet) for comparisons.

## Sweeps
- Sweeps must be restartable and not overwrite old results.
- Prefer deterministic naming: include key params + seed in output paths.

## Outputs
- Scripts should not “invent” metrics silently; coordinate new metrics with core + tests + docs.
