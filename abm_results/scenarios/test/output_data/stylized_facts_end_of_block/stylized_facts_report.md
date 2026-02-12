# Stylized Facts Report

## Executive summary
- Goal: first-pass econophysics diagnostics for market-like behavior in ABM-generated prices.
- Scope: fat tails, volatility clustering, leverage/asymmetry, and scaling across horizons.
- Inference level: diagnostic evidence only (not a formal proof or model validation test).

## Data and preprocessing
Input prices: `/home/danielemdn/Documents/repositories/ABM_Uni_v3/abm_results/scenarios/test/output_data/dex_price_end_of_block.npy`
Total sampled prices: **13000**
Missing/non-finite price share: **0.0000%**

## Method summary
- Horizons analyzed: 1, 5, 10, 20 steps.
- Critical note: for each horizon `h`, prices are first sub-sampled as `P[::h]`, then returns are computed as `Δlog(P_h)`.
- ACF diagnostics use `max_lag = min(120, n_h//4)` with horizon-specific sample sizes, and all ACF plots start at lag 1 (lag 0 omitted).
- Reported volatility-clustering lag cap in figures: **120**.
- Reported leverage lag cap in figures: **30**.

## Table 1. Per-horizon return summary
| Horizon | n_returns | mean | std | skewness | excess kurtosis |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 12999 | 3.516e-06 | 0.001443 | -0.2022 | 34.72 |
| 5 | 2599 | 1.734e-05 | 0.001922 | -0.2506 | 13.34 |
| 10 | 1299 | 3.337e-05 | 0.002225 | -0.542 | 3.228 |
| 20 | 649 | 6.735e-05 | 0.002932 | -0.3689 | 0.9458 |

## 1) Fat tails
| Horizon | excess kurtosis | abs(r)>3σ share | IQR outlier share | Hill α upper (1%,2%,5%) | Hill α lower (1%,2%,5%) |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 34.72 | 0.02069 | 0.07116 | 2.821, 1.99, 1.811 | 2.988, 2.532, 1.791 |
| 5 | 13.34 | 0.02001 | 0.01616 | 2.582, 2.304, 2.281 | 4.087, 3.538, 2.69 |
| 10 | 3.228 | 0.0154 | 0.004619 | 2.947, 2.947, 2.93 | 5.465, 5.465, 4.807 |
| 20 | 0.9458 | 0.01387 | 0 | 5.112, 5.112, 5.112 | 3.446, 3.446, 3.446 |
- Comment: fat tails are strong at short horizons and attenuate with aggregation, consistent with stylized-facts scaling behavior.
- Caveat: Hill estimates are threshold-sensitive and noisy in finite samples; use them as directional diagnostics.

## 2) Volatility clustering and leverage
| Horizon | ACF(abs(r), lag1) | ACF(r^2, lag1) | corr(r_t, abs(r_(t+1)) |
| --- | ---: | ---: | ---: |
| 1 | 0.4399 | 0.4799 | -0.07882 |
| 5 | 0.2969 | 0.4242 | -0.09668 |
| 10 | 0.1494 | 0.2717 | -0.06596 |
| 20 | 0.03883 | 0.05609 | -0.05587 |
- Comment: positive autocorrelation in volatility proxies indicates clustering; negative lagged return-volatility correlation is leverage-like asymmetry.
- Note: leverage magnitude is moderate and decreases with horizon.

## 3) Scaling / aggregation diagnostics
- Aggregation uses non-overlapping horizon-specific sampling: for each horizon h, we take prices P_h = P[::h], then compute one-step log-returns on P_h. This avoids overlapping-return dependence inflation but reduces sample size as h increases.
- KDEs are computed on standardized returns per horizon and compared against N(0,1) on log-y scale.
- Return ACFs are plotted jointly across horizons for short-memory diagnostics.
- Comment: non-overlapping sampling avoids overlap-induced dependence inflation at the cost of smaller effective sample size for large horizons.

## 4) Conclusion
- Overall: **largely consistent with common stylized facts** (fat tails + clustering present; leverage mixed/partial).
- These are diagnostics, not formal hypothesis tests.

## Figures
| Figure | HTML | PNG | PNG status |
| --- | --- | --- | --- |
| `fat_tails_qq_by_horizon` | `fat_tails_qq_by_horizon.html` | `fat_tails_qq_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `volatility_clustering_acf_by_horizon` | `volatility_clustering_acf_by_horizon.html` | `volatility_clustering_acf_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `leverage_lagged_correlation_by_horizon` | `leverage_lagged_correlation_by_horizon.html` | `leverage_lagged_correlation_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `scaling_kde_logy_by_horizon` | `scaling_kde_logy_by_horizon.html` | `scaling_kde_logy_by_horizon.png` | failed (Timed out during Plotly PNG export) |
| `return_acf_by_horizon` | `return_acf_by_horizon.html` | `return_acf_by_horizon.png` | failed (Timed out during Plotly PNG export) |