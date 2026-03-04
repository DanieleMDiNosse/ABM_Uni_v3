# Results provenance (cross-scenario summary figures)

This note records the provenance for the cross-scenario summary figures used in the Results section (hedged PnL heatmap, DEX share, fee levels, and fee value decomposition).

## Code provenance
- Git commit: `95b88ea3e838de999d0813b736b3d636b6fceb36`

## Experimental design (what is being compared)
- **Model variants**
  - Model 0: passive LPs only (no active LPs, no JIT).
  - Model 1: passive + active LPs (equal split).
  - Model 2: passive + active LPs + JIT liquidity provider (MEV searcher).
- **Fee schedules**: `static`, `toxicity`, `volatility_dex`, `volatility_cex`.
- **Replication**: 100 independent seeds per (model, fee schedule) point.

## Generation (how to reproduce)
- Entry point:
  - `python -m scripts.analysis.run_paper_figures`
- Base scenario:
  - `abm_results/scenarios/test.yml`
- Command (as used for the paper figures):
  - `python -m scripts.analysis.run_paper_figures --config abm_results/scenarios/test.yml --runs 100 --output-dir paper/images/analysis`

## Paper-stable artifacts
- PNG figures: `paper/images/analysis/png/`
  - `pnl_heatmap.png`
  - `dex_share_barplot.png`
  - `fee_value_barplot.png`
  - `mean_fee_barplot.png`
- Interactive (Plotly) versions: `paper/images/analysis/html/`
  - `pnl_heatmap.html`
  - `dex_share_barplot.html`
  - `fee_value_barplot.html`
  - `mean_fee_barplot.html`

## Notes
- Numerical values quoted in the manuscript for final hedged PnL correspond to the (mean ± sd) annotations embedded in `pnl_heatmap.html` and rendered in `pnl_heatmap.png`.
