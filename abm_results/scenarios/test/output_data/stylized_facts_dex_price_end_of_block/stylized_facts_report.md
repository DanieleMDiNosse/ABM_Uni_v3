# Stylized Facts Report

## Executive summary
- Goal: first-pass econophysics diagnostics for market-like behavior in ABM-generated series.
- Scope: fat tails, volatility clustering, leverage/asymmetry, scaling, and return autocorrelation.
- Inference level: diagnostics only (not proof of a generative model).

## Data provenance and preprocessing
- Input path: `/home/danielemdn/Documents/repositories/ABM_Uni_v3/abm_results/scenarios/test/output_data/dex_price_end_of_block.npy`
- Input kind: **prices**
- Return type analyzed: **log**
- Raw sample size: **13000**
- Dropped non-finite values: **0**
- Dropped non-positive prices (<=0): **0**
- Final clean sample size: **13000**
- Horizons analyzed: **1, 5, 10, 20** (non-overlapping construction).
- Construction rule: downsample prices at horizon h, then compute returns on sampled prices.

## Sample size table
| Horizon | n_returns | n_pos | n_neg | n_abs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12999 | 6236 | 5250 | 11486 |
| 5 | 2599 | 1437 | 1162 | 2599 |
| 10 | 1299 | 712 | 587 | 1299 |
| 20 | 649 | 338 | 311 | 649 |

## 1) Fat tails
- QQ plots vs Normal are qualitative diagnostics; tail uncertainty dominates finite samples.

| Horizon | excess kurtosis | abs(r)>3σ share | IQR outlier share | MAD-scale | q(0.99)/q(0.95) for abs(r) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.72 | 0.02069 | 0.07116 | 0.0004346 | 2.65 |
| 5 | 13.34 | 0.02001 | 0.01616 | 0.001275 | 1.968 |
| 10 | 3.228 | 0.0154 | 0.004619 | 0.001801 | 1.547 |
| 20 | 0.9458 | 0.01387 | 0 | 0.00273 | 1.658 |

### Hill tail index diagnostics (alpha = 1/gamma)
| Horizon | Tail side | n_tail | k-range | k* | alpha(k*) [95% CI] | gamma(k*) [95% CI] | alpha range (middle 50% k-grid) |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| 1 | upper | 6236 | 10..1000 | 510 | 1.675 [1.529, 1.82] | 0.5972 [0.5454, 0.649] | [1.651, 1.832] |
| 1 | lower | 5250 | 10..1000 | 360 | 1.784 [1.6, 1.968] | 0.5605 [0.5026, 0.6184] | [1.617, 1.824] |
| 1 | abs | 11486 | 10..1000 | 560 | 1.803 [1.654, 1.952] | 0.5547 [0.5087, 0.6006] | [1.783, 2.12] |
| 5 | upper | 1437 | 10..718 | 188 | 2.42 [2.074, 2.766] | 0.4132 [0.3542, 0.4723] | [1.809, 2.42] |
| 5 | lower | 1162 | 10..581 | 188 | 2.114 [1.812, 2.416] | 0.473 [0.4054, 0.5407] | [1.539, 2.15] |
| 5 | abs | 2599 | 10..1000 | 270 | 2.193 [1.932, 2.455] | 0.4559 [0.4015, 0.5103] | [1.837, 2.22] |
| 10 | upper | 712 | 10..356 | 128 | 2.779 [2.298, 3.26] | 0.3598 [0.2975, 0.4222] | [2.27, 3.034] |
| 10 | lower | 587 | 10..293 | 201 | 1.812 [1.561, 2.062] | 0.552 [0.4757, 0.6283] | [1.772, 2.668] |
| 10 | abs | 1299 | 10..649 | 319 | 2.344 [2.087, 2.601] | 0.4266 [0.3798, 0.4734] | [1.979, 2.641] |
| 20 | upper | 338 | 10..169 | 85 | 2.833 [2.23, 3.435] | 0.353 [0.278, 0.4281] | [2.415, 4.107] |
| 20 | lower | 311 | 10..155 | 94 | 2.039 [1.627, 2.451] | 0.4905 [0.3913, 0.5896] | [1.916, 3.336] |
| 20 | abs | 649 | 10..324 | 197 | 2.344 [2.017, 2.671] | 0.4266 [0.3671, 0.4862] | [2.129, 3.443] |
- Hill warning: for light-tailed data Hill can imply gamma≈0 and alpha→infinity, which is misleading.
- Hill warning: returns are heteroskedastic; i.i.d. tail assumptions are violated by volatility clustering.
- Use Hill plots as stability diagnostics, not as single-point proof.

## 2) Volatility clustering
- ACF lag cap uses default rule `min(250, floor(n/10))` with chosen `max_lag_acf=64`.

| Horizon | ACF(abs(r), lag1) | ACF(r^2, lag1) |
| --- | ---: | ---: |
| 1 | 0.4399 | 0.4799 |
| 5 | 0.2969 | 0.4242 |
| 10 | 0.1494 | 0.2717 |
| 20 | 0.03883 | 0.05609 |
- Slow decay in ACF(abs(r)) and ACF(r^2) is consistent with clustering / long-memory proxies.
- Optional GARCH(1,1) fit skipped in this first-pass report.

## 3) Leverage and asymmetry
- Lagged correlations computed for lags 1..30.

| Horizon | corr($r_t, abs(r_{t+1}$) | corr($r_t, r_{t+1}^2$) |
| --- | ---: | ---: |
| 1 | -0.07882 | -0.08732 |
| 5 | -0.09668 | -0.1421 |
| 10 | -0.06596 | -0.1017 |
| 20 | -0.05587 | -0.05252 |
- Negative values are consistent with leverage effect (common in equities, not universal).

## 4) Scaling / aggregation
- Distributions are compared across horizons using non-overlapping construction only.
- KDEs are shown on standardized returns with log-y scale and Gaussian reference.
- Caveat: KDE bandwidth and tail sparsity affect large-horizon interpretation.

## 5) Return autocorrelation
- Return ACF is reported by horizon, starting at lag 1 (lag 0 removed).
- Near-zero short-lag ACF is typical for liquid daily data; microstructure can alter this at high frequency.

## 6) Conclusion
- Verdict: **Consistent with stylized facts** (diagnostic score = 12/12).
- Plausible deviation sources: microstructure noise, illiquidity, regime shifts, sampling artifacts, or missing-data handling.

## Limitations
- Diagnostics are descriptive, not formal hypothesis tests.
- Hill estimates are sensitive to threshold choice and dependence in returns.
- Finite-sample uncertainty is substantial at large horizons and in one-sided tails.

## What I would do next
1. Add threshold-stability checks with alternative EVT estimators (Pickands, moment).
2. Repeat tail analysis after volatility normalization / declustering.
3. Check sub-sample stability across regimes and simulation seeds.
4. Compare non-overlapping vs overlapping horizons as a sensitivity analysis.

## Generated tables
- Horizon summary: `/home/danielemdn/Documents/repositories/ABM_Uni_v3/abm_results/scenarios/test/output_data/stylized_facts_dex_price_end_of_block/tables/horizon_summary.csv`
- Tail metrics: `/home/danielemdn/Documents/repositories/ABM_Uni_v3/abm_results/scenarios/test/output_data/stylized_facts_dex_price_end_of_block/tables/tail_metrics.csv`
- Hill working estimates: `/home/danielemdn/Documents/repositories/ABM_Uni_v3/abm_results/scenarios/test/output_data/stylized_facts_dex_price_end_of_block/tables/hill_working_estimates.csv`

## Figures
| Figure | HTML | PNG | PNG status |
| --- | --- | --- | --- |
| `fat_tails_qq_by_horizon` | `fat_tails_qq_by_horizon.html` | `fat_tails_qq_by_horizon.png` | failed (No local Chrome/Chromium executable found; PNG export skipped.) |
| `volatility_clustering_acf_by_horizon` | `volatility_clustering_acf_by_horizon.html` | `volatility_clustering_acf_by_horizon.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `leverage_lagged_correlation_by_horizon` | `leverage_lagged_correlation_by_horizon.html` | `leverage_lagged_correlation_by_horizon.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `scaling_kde_logy_by_horizon` | `scaling_kde_logy_by_horizon.html` | `scaling_kde_logy_by_horizon.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `return_acf_by_horizon` | `return_acf_by_horizon.html` | `return_acf_by_horizon.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_upper_h1` | `hill_plot_upper_h1.html` | `hill_plot_upper_h1.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_lower_h1` | `hill_plot_lower_h1.html` | `hill_plot_lower_h1.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_abs_h1` | `hill_plot_abs_h1.html` | `hill_plot_abs_h1.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_upper_h5` | `hill_plot_upper_h5.html` | `hill_plot_upper_h5.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_lower_h5` | `hill_plot_lower_h5.html` | `hill_plot_lower_h5.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_abs_h5` | `hill_plot_abs_h5.html` | `hill_plot_abs_h5.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_upper_h10` | `hill_plot_upper_h10.html` | `hill_plot_upper_h10.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_lower_h10` | `hill_plot_lower_h10.html` | `hill_plot_lower_h10.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_abs_h10` | `hill_plot_abs_h10.html` | `hill_plot_abs_h10.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_upper_h20` | `hill_plot_upper_h20.html` | `hill_plot_upper_h20.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_lower_h20` | `hill_plot_lower_h20.html` | `hill_plot_lower_h20.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |
| `hill_plot_abs_h20` | `hill_plot_abs_h20.html` | `hill_plot_abs_h20.png` | failed (PNG export disabled after first failure: No local Chrome/Chromium executable found; PNG export skipped.) |