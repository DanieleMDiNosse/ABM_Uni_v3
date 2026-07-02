# AGENTS.md — `paper/` (Manuscript & Paper-Stable Artifacts)

This folder contains the LaTeX manuscript(s) and a **snapshot** of the exact figures used in the paper. Treat it as a reproducible research product: the manuscript must remain scientifically defensible, traceable, and rebuildable from versioned sources.

## Mission (paper-grade)
- Write in a clear, scientific tone; develop intuition before formalism.
- Make every claim falsifiable: each number/figure must be traceable to a stored artifact and a reproducible run.
- Prefer conservative, model-conditional language (simulation evidence is not empirical truth).

## Folder map (current)
- Manuscripts:
  - `paper/ABM_paper.tex` (main paper)
  - `paper/extended_abstract.tex` (extended abstract)
- Bibliography:
  - `paper/bibliography.bib`
- Paper-stable figure snapshots (LaTeX should reference these):
  - `paper/images/` (PNG/PDF)
  - `paper/figures_st_app/` (appendix PDFs)
- Generated pipeline outputs (source for regeneration; not paper-stable):
  - `abm_results/scenarios/<scenario_name>/runs/<run_id>/{png,html,output_data,logs}/`

If you add a new manuscript entrypoint, document it here and keep build commands updated.

## Build the PDF (avoid polluting repo root)
Run from repo root:
- `latexmk -cd -pdf paper/ABM_paper.tex`
- `latexmk -cd -pdf paper/extended_abstract.tex`

Why `-cd`: it keeps build artifacts under `paper/` instead of writing `.aux/.log/...` in the repo root.

Optional cleanup:
- `latexmk -cd -c paper/ABM_paper.tex` (remove aux files, keep PDF)
- `latexmk -cd -C paper/ABM_paper.tex` (deep clean; may remove PDF depending on latexmk version)

## Figure policy (snapshot is the LaTeX source of truth)
Two figure locations exist:
- Generated outputs (not paper-stable): `abm_results/scenarios/.../runs/.../png`
- Paper-stable snapshots (LaTeX should reference these): `paper/images/` and `paper/figures_st_app/`

### Required sync rule for paper updates
When results are regenerated:
1. Run the simulator/analysis from an explicit scenario YAML + seed(s).
2. Identify which generated outputs correspond to each paper figure/table.
3. Copy only publication-ready artifacts into `paper/images/...` (or `paper/figures_st_app/...`).
4. Commit the snapshots alongside the manuscript changes.

Do not manually edit generated images or hand-type summary statistics into LaTeX without saving the reproducible table/artifact that produced them.

## Backend-to-paper traceability contract (non-negotiable)
Every claim, number, and figure in `paper/*.tex` must be traceable to:
- a stored artifact: `paper/images/...` (figures) and/or a stored machine-readable table under `abm_results/.../output_data/...`, and
- a generating script + configuration + run provenance.

Minimum provenance to record per figure/table (store as a short note near the figure/table in LaTeX, or in a small manifest file in `paper/`):
- Source script(s) (e.g., `scripts/run.py`, `scripts/stylized_facts_report.py`)
- Scenario YAML path (e.g., `configs/scenarios/<name>.yml`) and the run id folder
- Seed(s) and any inference settings (bootstraps, aggregation windows, sampling frequency)
- Code provenance: git commit hash (if available) and the exact command line used

## Paper-grade statistical defensibility checklist (simulation)
Before finalizing an updated section/claim:
- Replication: avoid single-run conclusions; report how many independent seeds were used.
- Uncertainty: report variability across runs (e.g., quantiles/CI) for key effects, not only point estimates.
- Dependence: when using within-run time series, account for autocorrelation (e.g., block bootstrap / clustered SE).
- Robustness: show at least one sensitivity check for central claims (parameters, agent mix, fee controller variant).
- Baselines: include and clearly label counterfactuals (e.g., static fees vs dynamic fees; no-JIT vs JIT).
- Units/numéraire: keep valuation conventions explicit and consistent (token0/token1, token1 numéraire, etc.).
- Interpretation: keep statements model-conditional; separate mechanisms (what the ABM enforces) from outcomes (what emerges).

## Writing style (clarity first)
- Open each technical subsection with: **Question → mechanism → measurement → finding**.
- Define notation on first use and keep it consistent with the codebase (price vs sqrt-price, ticks, fees).
- Explain *why* a modeling choice is made (what it approximates, what it ignores, and expected failure modes).
- Every figure caption must state: setting (scenario/controller), metric, and takeaway in one paragraph.
- Avoid TODO placeholders in manuscript text; either remove them or label as future work/limitations explicitly.

## What NOT to do
- Don’t change simulator behavior to “fit” the narrative without also adding diagnostics/tests (follow repo-root `AGENTS.md`).
- Don’t overwrite paper-stable snapshots without updating provenance and updating the manuscript accordingly.
- Don’t present descriptive plots as causal claims; qualify conclusions and state assumptions.
