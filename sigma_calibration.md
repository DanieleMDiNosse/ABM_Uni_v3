# Calibrating `cex_sigma` from Binance ETH/USDC Data

This note explains how to calibrate the CEX volatility parameter `cex_sigma`
in the ABM simulation using Binance ETH/USDC 1‑second data over (roughly)
two years. The key assumptions are:

- The dataset consists of 1‑second candles with columns like  
  `['Open time', 'Open', 'High', 'Low', 'Close', ...]`.
- One **micro step** in the simulator corresponds to **1 second** of real time.
- The CEX mid‑price follows a geometric Brownian motion (GBM) with
  per‑second parameters.

The goal is to derive **low** and **high** values of `cex_sigma` that are
empirically grounded in the historical ETH/USDC volatility.

---

## Reference Implementation (`sigma_calibration.py`)

The full workflow below is implemented in `sigma_calibration.py`. Run it on
your raw Binance CSV to produce the 1-second volatility series, percentile
table, and low/high regime medians:

```bash
python sigma_calibration.py /path/to/ETHUSDC_1s.csv \
    --window-seconds 300 \
    --low-quantile 0.20 \
    --high-quantile 0.80 \
    --save-csv abm_results/ethusdc_sigma_1s.csv
```

Key options:

- `--window-seconds`: length of the time-based rolling window (defaults to
  5 minutes). Increase it for smoother volatility estimates.
- `--low-quantile` / `--high-quantile`: define the regime thresholds used
  to extract representative `cex_sigma` values.
- `--save-csv` / `--save-parquet`: optionally persist the time series with
  columns `close`, `log_return_1s`, `sigma_1s`, and `sigma_annualized`.

Internally the script mirrors every mathematical step outlined in the
sections below (timestamp parsing, log-return computation, rolling
realized volatility, quantile extraction).

---

## 1. How `cex_sigma` Enters the Model

In `utils.py`, the reference CEX is modeled as:

```python
class ReferenceMarket:
    m: float  # CEX price of token A in token B
    mu: float # drift (per step) of log-returns
    sigma: float  # vol (per step) of log-returns
    ...

    def diffuse_only(self) -> float:
        z = np.random.normal()
        self.m *= math.exp(self.mu - 0.5 * self.sigma + self.sigma * z)
```

If we denote the price at (discrete) time step \( t \) by \( m_t \), then

\[
\log m_{t+1} - \log m_t
  = \mu - \tfrac{1}{2}\sigma + \sigma Z_t, \quad Z_t \sim \mathcal{N}(0,1)
\]

where:

- \( \mu \) is the **drift per step** (here, per second),
- \( \sigma \) is the **volatility per step** of log‑returns (here, per second).

Since one micro‑step is **1 second**, we interpret `cex_sigma` as:

\[
\sigma_{\text{1s}} := \text{standard deviation of 1‑second log returns}.
\]

This is exactly what we should estimate from the Binance 1‑second ETH/USDC
time series and then feed into the simulator.

---

## 2. Preparing the Binance 1‑Second Data

Assume the raw CSV contains at least:

- `Open time` (usually a Unix timestamp in **milliseconds**),
- `Close` (close price of the 1‑second candle, in USDC per ETH).

### 2.1. Time Index and Sorting

Let \( P_t \) be the close price at second \( t \). In pandas:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("ETHUSDC_1s_binance.csv")

# Convert Open time to UTC datetime index
if np.issubdtype(df['Open time'].dtype, np.number):
    ts = pd.to_datetime(df['Open time'], unit='ms', utc=True)
else:
    ts = pd.to_datetime(df['Open time'], utc=True)

df = (
    df.assign(timestamp=ts)
      .sort_values('timestamp')
      .set_index('timestamp')
)

df['Close'] = df['Close'].astype(float)
```

This gives a time‑indexed series \( P_t = \text{Close}_t \) at (approximately)
1‑second intervals. Minor gaps (missing seconds) are acceptable; log‑returns
are only computed where both \( P_{t-1} \) and \( P_t \) exist.

### 2.2. Log‑Returns at 1‑Second Frequency

Define the **1‑second log‑return**:

\[
r_t := \log P_t - \log P_{t-1}
     = \log \left( \frac{P_t}{P_{t-1}} \right).
\]

In pandas:

```python
log_price = np.log(df['Close'])
log_ret_1s = log_price.diff()
log_ret_1s = log_ret_1s.replace([np.inf, -np.inf], np.nan)
```

The sequence \( \{ r_t \} \) is the empirical counterpart of the GBM
log‑returns driven by `mu` and `sigma` in the simulator.

---

## 3. Estimating Per‑Second Volatility

We want an estimate of the **instantaneous per‑second volatility**
\( \sigma_{\text{1s}} \), not a multi‑hour/day volatility. A standard
approach is to use a **rolling window** of recent log‑returns.

### 3.1. Rolling Realized Variance

Fix a window length \( N \) in seconds (e.g. \( N = 300 \) for 5 minutes,
or \( N = 900 \) for 15 minutes). For each time \( t \ge N \), define the
rolling empirical variance over the last \( N \) returns:

\[
\hat{\sigma}^2_{t,\text{1s}}
  := \frac{1}{N - 1}
     \sum_{i=t-N+1}^{t}
     \left( r_i - \bar{r}_{t} \right)^2,
\]

where

\[
\bar{r}_{t} = \frac{1}{N} \sum_{i=t-N+1}^{t} r_i
\]

is the sample mean over the window. The corresponding rolling standard
deviation is:

\[
\hat{\sigma}_{t,\text{1s}} := \sqrt{\hat{\sigma}^2_{t,\text{1s}}}.
\]

Under the GBM model with small drift, this rolling standard deviation is
an estimator of the **per‑second volatility** \( \sigma_{\text{1s}} \).

In pandas, using a time‑based rolling window:

```python
window_seconds = 300  # e.g. 5-minute window

sigma_1s = (
    log_ret_1s
    .rolling(f"{window_seconds}s")
    .std()
)
```

This produces a **time series** \( \{ \hat{\sigma}_{t,\text{1s}} \} \),
each value being an estimate of the per‑second volatility at time \( t \),
based on the recent window of returns.

### 3.2. Link to Annualized Volatility (Optional)

Sometimes it is convenient to express volatility in **annualized** units.
If:

- \( \hat{\sigma}_{t,\text{1s}} \) is the per‑second volatility estimate,
- there are \( S_{\text{year}} = 365 \cdot 24 \cdot 60 \cdot 60 \) seconds
  per year,

then the corresponding **annualized volatility** is:

\[
\hat{\sigma}_{t,\text{ann}}
  = \hat{\sigma}_{t,\text{1s}} \sqrt{S_{\text{year}}}.
\]

In pandas:

```python
seconds_per_year = 365 * 24 * 60 * 60
sigma_annualized = sigma_1s * np.sqrt(seconds_per_year)
```

**Important:** in the simulator, `cex_sigma` is *per micro‑step* (per
second), so you should feed the **per‑second** value \( \hat{\sigma}_{t,\text{1s}} \)
directly, not the annualized one.

---

## 4. Using the 2‑Year Dataset to Define “Low” and “High” Volatility

Given the 2‑year series of 1‑second returns \( \{ r_t \} \), and the derived
rolling volatility \( \{ \hat{\sigma}_{t,\text{1s}} \} \), we can use
**quantiles** of this volatility series to define empirical volatility
regimes.

### 4.1. Build the Volatility Series and Clean It

Putting the steps together:

```python
vol_df = pd.DataFrame({
    "log_return_1s": log_ret_1s,
    "sigma_1s": sigma_1s,
})

vol_df = vol_df.dropna(subset=["sigma_1s"])
```

Now `vol_df["sigma_1s"]` contains a long time series of per‑second
volatility estimates across the entire 2‑year period.

### 4.2. Inspect the Distribution of Per‑Second Volatility

We treat each volatility estimate in `sigma_1s` as a sample from the
underlying **distribution of instantaneous per‑second volatility**. Its
quantiles describe “typical” vs “extreme” volatility environments.

```python
q = vol_df["sigma_1s"].quantile(
    [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
)
print(q)
```

Conceptually, if we denote the random per‑second volatility by
\( \Sigma_{\text{1s}} \), the empirical quantile function \( Q(p) \)
such that

\[
Q(p) \approx \inf \{ x : \mathbb{P}(\Sigma_{\text{1s}} \le x) \ge p \},
\]

can be estimated from the sample as:

\[
Q(p) \approx \text{quantile}_p(\{\hat{\sigma}_{t,\text{1s}}\}_t).
\]

This lets us define volatility regimes in terms of **probability mass**
over the 2‑year historical period.

### 4.3. Defining “Low” and “High” Regimes

A simple and robust regime definition is:

- **Low volatility regime:**  
  per‑second volatility below a lower quantile \( Q(p_{\text{low}}) \),
  e.g. \( p_{\text{low}} = 0.2 \) (20th percentile).
- **High volatility regime:**  
  per‑second volatility above an upper quantile \( Q(p_{\text{high}}) \),
  e.g. \( p_{\text{high}} = 0.8 \) (80th percentile).

Formally:

\[
\text{low‑vol regime}  := \{\hat{\sigma}_{t,\text{1s}} \le Q(0.2)\}, \\
\text{high‑vol regime} := \{\hat{\sigma}_{t,\text{1s}} \ge Q(0.8)\}.
\]

You can pick slightly different quantiles (e.g. 10% / 90%, or 25% / 75%)
depending on how “extreme” you want the regimes to be.

### 4.4. Mapping Regimes to Scenario Parameters

For each regime, we want a **single scalar** `cex_sigma` to plug into the
YAML scenarios. A natural choice is to take the **median** of the
volatility estimates within the regime:

- Let
  \[
    \mathcal{S}_{\text{low}}
      := \{ \hat{\sigma}_{t,\text{1s}} : \hat{\sigma}_{t,\text{1s}} \le Q(p_{\text{low}}) \},
  \]
  and define
  \[
    \sigma_{\text{1s, low}} := \text{median}(\mathcal{S}_{\text{low}}).
  \]

- Let
  \[
    \mathcal{S}_{\text{high}}
      := \{ \hat{\sigma}_{t,\text{1s}} : \hat{\sigma}_{t,\text{1s}} \ge Q(p_{\text{high}}) \},
  \]
  and define
  \[
    \sigma_{\text{1s, high}} := \text{median}(\mathcal{S}_{\text{high}}).
  \]

In pandas:

```python
sigma_series = vol_df["sigma_1s"]

p_low, p_high = 0.2, 0.8
q_low, q_high = sigma_series.quantile([p_low, p_high])

sigma_1s_low  = sigma_series[sigma_series <= q_low].median()
sigma_1s_high = sigma_series[sigma_series >= q_high].median()

print("Low-vol cex_sigma (per second):", sigma_1s_low)
print("High-vol cex_sigma (per second):", sigma_1s_high)
```

These two numbers are your **empirically calibrated** per‑second
volatility levels for “low” and “high” volatility regimes, based on the
full 2‑year ETH/USDC dataset.

---

## 5. Plugging the Calibrated Values into the ABM

Once you have `sigma_1s_low` and `sigma_1s_high`:

- Use `sigma_1s_low` as `cex_sigma` in `scenarios/low_volatility.yml`.
- Use `sigma_1s_high` as `cex_sigma` in `scenarios/high_volatility.yml`.

For example:

```yaml
# scenarios/low_volatility.yml
simulate:
  cex_sigma: <sigma_1s_low from calibration>   # per-second volatility

# scenarios/high_volatility.yml
simulate:
  cex_sigma: <sigma_1s_high from calibration>  # per-second volatility
```

Because one micro step equals one second in the simulation, **no further
time‑scaling is required**: the calibrated `sigma_1s_*` values can be
used directly as `cex_sigma`.

---

## 6. Practical Notes for a 2‑Year, 1‑Second Dataset

- **Memory considerations:** 2 years of 1‑second data is on the order of
  \( 2 \times 365 \times 24 \times 3600 \approx 63 \times 10^6 \) rows.
  If loading the full dataset at once is heavy, you can:
  - Process data in chunks, accumulating the distribution of
    `sigma_1s` (e.g. storing quantile summaries or random subsamples).
  - Restrict to trading hours of interest (e.g. only certain periods).

- **Choice of window length:**  
  Short windows (e.g. 5 minutes) capture very fast volatility changes but
  yield noisier estimates. Longer windows (e.g. 15–60 minutes) give
  smoother but slower‑responding volatility. The choice depends on how
  quickly you want the CEX environment to change in the ABM.

- **Robustness:**  
  You may want to winsorize or clip extreme outliers in `sigma_1s`
  (e.g. due to bad ticks or erroneous prices) before computing quantiles.

By following this pipeline on your actual Binance ETH/USDC 1‑second data,
you obtain a principled, data‑driven calibration of `cex_sigma` for both
low‑ and high‑volatility simulation scenarios, fully consistent with the
GBM dynamics implemented in the ABM.
