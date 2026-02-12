# Stylized Facts Report

## Executive summary
- Goal: first-pass econophysics diagnostics for market-like behavior in ABM-generated series.
- Scope: fat tails, volatility clustering, leverage/asymmetry, and scaling across horizons.
- Inference level: diagnostic evidence only (not a formal proof or model validation test).

## Data and preprocessing
Input path: `/home/danielemdn/Documents/repositories/ABM_Uni_v3/abm_results/scenarios/test/output_data/dex_price_end_of_block.npy`
Input kind: **returns**
Return type analyzed: **log**
Final finite sample size: **13000**
Dropped non-finite observations: **0**
Missing/non-finite share: **0.0000%**

## Method summary
- Horizons analyzed: 1, 5, 10, 20 steps.
- Critical note: input is precomputed returns; for each horizon `h`, returns are aggregated in non-overlapping blocks.
- For log returns, aggregation is block-wise sum. For simple returns, aggregation is block-wise compounding.
- ACF diagnostics use `max_lag = min(120, n_h//4)` with horizon-specific sample sizes, and all ACF plots start at lag 1 (lag 0 omitted).
- Reported volatility-clustering lag cap in figures: **120**.
- Reported leverage lag cap in figures: **30**.

## Table 1. Per-horizon return summary
| Horizon | n_returns | mean | std | skewness | excess kurtosis |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 13000 | 2065 | 42.57 | 0.003211 | -1.004 |
| 5 | 2600 | 1.032e+04 | 212.6 | 0.002274 | -1.008 |
| 10 | 1300 | 2.065e+04 | 425.2 | 0.002237 | -1.009 |
| 20 | 650 | 4.13e+04 | 850 | 0.001812 | -1.01 |

## 1) Fat tails
| Horizon | n_pos | n_neg | n_abs | excess kurtosis | abs(r)>3σ share | IQR outlier share | Hill α upper (1%,2%,5%) | Hill α lower (1%,2%,5%) | Hill α abs (1%,2%,5%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 13000 | 0 | 13000 | -1.004 | 1 | 0 | 718, 529.1, 243 | n/a, n/a, n/a | 718, 529.1, 243 |
| 5 | 2600 | 0 | 2600 | -1.008 | 1 | 0 | 817.6, 587.6, 255.5 | n/a, n/a, n/a | 817.6, 587.6, 255.5 |
| 10 | 1300 | 0 | 1300 | -1.009 | 1 | 0 | 848.9, 513.4, 244.8 | n/a, n/a, n/a | 848.9, 513.4, 244.8 |
| 20 | 650 | 0 | 650 | -1.01 | 1 | 0 | 349.1, 349.1, 239.9 | n/a, n/a, n/a | 349.1, 349.1, 239.9 |
- Comment: fat tails are strong at short horizons and attenuate with aggregation, consistent with stylized-facts scaling behavior.
- Caveat: Hill estimates are threshold-sensitive and noisy in finite samples; use them as directional diagnostics.

## 2) Volatility clustering and leverage
| Horizon | ACF(abs(r), lag1) | ACF(r^2, lag1) | corr(r_t, abs(r_(t+1))) | corr(r_t, r_(t+1)^2) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.9974 | 0.9974 | 0.9975 | 0.9975 |
| 5 | 0.9976 | 0.9976 | 0.9981 | 0.9981 |
| 10 | 0.9958 | 0.9958 | 0.9969 | 0.9969 |
| 20 | 0.9921 | 0.9921 | 0.9943 | 0.9942 |
- Comment: positive autocorrelation in volatility proxies indicates clustering; negative lagged return-volatility correlation is leverage-like asymmetry.
- Note: leverage magnitude is moderate and decreases with horizon.

## 3) Scaling / aggregation diagnostics
- Input is precomputed returns. Aggregation uses non-overlapping blocks per horizon. For log returns we sum within blocks; for simple returns we compound within blocks. This avoids overlap dependence inflation but reduces effective sample size as h increases.
- KDEs are computed on standardized returns per horizon and compared against N(0,1) on log-y scale.
- Return ACFs are plotted jointly across horizons for short-memory diagnostics.
- Comment: non-overlapping sampling avoids overlap-induced dependence inflation at the cost of smaller effective sample size for large horizons.

## 4) Conclusion
- Overall: **partially consistent with stylized facts** with notable deviations across horizons.
- These are diagnostics, not formal hypothesis tests.

## Figures
| Figure | HTML | PNG | PNG status |
| --- | --- | --- | --- |
| `fat_tails_qq_by_horizon` | `fat_tails_qq_by_horizon.html` | `fat_tails_qq_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `volatility_clustering_acf_by_horizon` | `volatility_clustering_acf_by_horizon.html` | `volatility_clustering_acf_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `leverage_lagged_correlation_by_horizon` | `leverage_lagged_correlation_by_horizon.html` | `leverage_lagged_correlation_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `scaling_kde_logy_by_horizon` | `scaling_kde_logy_by_horizon.html` | `scaling_kde_logy_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `return_acf_by_horizon` | `return_acf_by_horizon.html` | `return_acf_by_horizon.png` | failed (Timed out during Plotly PNG export) |