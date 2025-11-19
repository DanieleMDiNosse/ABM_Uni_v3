#!/usr/bin/env python3
"""
Compute per-second realized volatility from Binance ETH/USDC 1-second data.

This script implements the workflow described in sigma_calibration.md:
1. Load Binance-style CSV data with an "Open time" column and "Close" price.
2. Convert the timestamp to UTC datetimes and sort chronologically.
3. Compute 1-second log returns.
4. Estimate per-second volatility via a rolling standard deviation.
5. Report quantiles of the distribution and regime-specific medians.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate per-second volatility (cex_sigma) from 1s Binance data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to the Binance 1-second dataset (CSV, Parquet, or pickle).",
    )
    parser.add_argument(
        "--time-column",
        default="Open time",
        help="Column containing the 1-second timestamp.",
    )
    parser.add_argument(
        "--close-column",
        default="Close",
        help="Column containing the close price (quoted in USDC per ETH).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=600,
        help="Length of the rolling window (in seconds) for realized volatility.",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=0.2,
        help="Quantile used to define the low-volatility regime threshold.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.8,
        help="Quantile used to define the high-volatility regime threshold.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
        help="Percentiles to report for the sigma distribution.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        help="Optional path to write a CSV with timestamp, close, log_return_1s, sigma_1s, sigma_annualized.",
    )
    parser.add_argument(
        "--save-parquet",
        type=Path,
        help="Optional path to write the same output dataframe as Parquet.",
    )
    return parser.parse_args()


def load_binance_series(
    data_path: Path,
    time_column: str,
    close_column: str,
) -> pd.Series:
    required_cols: Sequence[str] = (time_column, close_column)
    df = _load_input_frame(data_path, required_cols)

    time_series = df[time_column]
    ts = _parse_timestamp(time_series)

    close = pd.to_numeric(df[close_column], errors="coerce")

    cleaned = (
        pd.DataFrame({"timestamp": ts, "close": close})
        .dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    if cleaned.empty:
        raise ValueError("Dataset is empty after cleaning timestamps/prices.")

    return cleaned["close"]


def _load_input_frame(path: Path, required_cols: Sequence[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, usecols=required_cols)
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(
            f"Unsupported data format '{path.suffix}'. Use CSV, Parquet, or pickle."
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Loaded object from {path} is not a pandas DataFrame.")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")

    return df.loc[:, list(required_cols)]


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Convert the Binance Open time column to UTC datetimes."""
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    sample = non_null.iloc[0]
    if is_numeric_dtype(series):
        unit = "ms" if float(sample) > 1e12 else "s"
        return pd.to_datetime(series, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def compute_log_returns(close: pd.Series) -> pd.Series:
    log_price = np.log(close)
    log_ret = log_price.diff()
    return log_ret.replace([np.inf, -np.inf], np.nan)


def compute_sigma_1s(log_ret: pd.Series, window_seconds: int) -> pd.Series:
    if window_seconds < 2:
        raise ValueError("window_seconds must be at least 2.")
    return log_ret.rolling(f"{window_seconds}s").std()


@dataclass
class RegimeSummary:
    percentiles: pd.Series
    sigma_low: float
    sigma_high: float
    threshold_low: float
    threshold_high: float


def summarize_sigma(
    sigma_series: pd.Series,
    percentiles: Iterable[float],
    low_quantile: float,
    high_quantile: float,
) -> RegimeSummary:
    cleaned = sigma_series.dropna()
    if cleaned.empty:
        raise ValueError("No rolling volatility values available. Check the window size.")

    percentiles_series = cleaned.quantile(percentiles)
    q_low = cleaned.quantile(low_quantile)
    q_high = cleaned.quantile(high_quantile)

    sigma_low = cleaned[cleaned <= q_low].median()
    sigma_high = cleaned[cleaned >= q_high].median()

    return RegimeSummary(
        percentiles=percentiles_series,
        sigma_low=float(sigma_low),
        sigma_high=float(sigma_high),
        threshold_low=float(q_low),
        threshold_high=float(q_high),
    )


def main() -> None:
    args = parse_args()

    close = load_binance_series(args.data_path, args.time_column, args.close_column)
    log_ret = compute_log_returns(close)
    sigma_1s = compute_sigma_1s(log_ret, args.window_seconds)
    sigma_ann = sigma_1s * np.sqrt(SECONDS_PER_YEAR)

    summary = summarize_sigma(
        sigma_1s,
        percentiles=args.percentiles,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )

    print(f"Loaded {len(close):,} rows from {args.data_path}")
    print(f"Rolling window: {args.window_seconds} seconds ({args.window_seconds/60:.2f} minutes)")
    print("\nPer-second sigma percentiles:")
    for p, value in summary.percentiles.items():
        print(f"  p={p:>5.2%}: sigma_1s={value:.6e}")

    print("\nRegime thresholds:")
    print(f"  low regime threshold (quantile {args.low_quantile:.2%}): {summary.threshold_low:.6e}")
    print(f"  high regime threshold (quantile {args.high_quantile:.2%}): {summary.threshold_high:.6e}")
    print(f"  low regime median sigma_1s:  {summary.sigma_low:.6e}")
    print(f"  high regime median sigma_1s: {summary.sigma_high:.6e}")
    print("  (Use these per-second sigmas directly as `cex_sigma`.)")

    if args.save_csv or args.save_parquet:
        out_df = pd.DataFrame(
            {
                "close": close,
                "log_return_1s": log_ret,
                "sigma_1s": sigma_1s,
                "sigma_annualized": sigma_ann,
            }
        )
        if args.save_csv:
            out_csv = args.save_csv
            out_df.to_csv(out_csv, index=True)
            print(f"Wrote volatility series to {out_csv}")
        if args.save_parquet:
            out_parquet = args.save_parquet
            out_df.to_parquet(out_parquet, index=True)
            print(f"Wrote volatility series to {out_parquet}")


if __name__ == "__main__":
    main()
