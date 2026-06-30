from __future__ import annotations

import numpy as np

import abm_webapp.app as webapp_app


def test_downsample_indices_respects_bounds_and_endpoints() -> None:
    idx = webapp_app._downsample_indices(13_000, max_points=1_800)
    assert idx.size <= 1_800
    assert int(idx[0]) == 0
    assert int(idx[-1]) == 12_999
    assert np.all(np.diff(idx) > 0)


def test_price_figure_trace_length_is_capped() -> None:
    rows = []
    for t in range(13_000):
        rows.append(
            dict(
                t=t,
                dex_price=1.0 + 1e-5 * t,
                cex_price=1.0 + 1.2e-5 * t,
                band_lo=0.99,
                band_hi=1.01,
            )
        )
    fig = webapp_app._build_price_figure(rows)
    # The band and price traces should use downsampled time-series points.
    lengths = [len(trace.x) for trace in fig.data]
    assert all(length <= webapp_app.MAX_TIMESERIES_POINTS for length in lengths)


def test_live_metrics_cache_keeps_full_history_by_default() -> None:
    webapp_app._clear_metrics_cache()
    rows = [dict(t=t, dex_price=1.0 + 1e-5 * t) for t in range(13_000)]

    stored_rows, _version = webapp_app._set_cached_metrics("full-history-run", rows)

    assert len(stored_rows) == 13_000
    assert stored_rows[0]["t"] == 0
    assert stored_rows[-1]["t"] == 12_999


def test_live_metrics_cache_limit_is_applied_when_configured(monkeypatch) -> None:
    webapp_app._clear_metrics_cache()
    monkeypatch.setattr(webapp_app, "LIVE_METRICS_LIMIT", 100)
    rows = [dict(t=t, dex_price=1.0 + 1e-5 * t) for t in range(250)]

    stored_rows, _version = webapp_app._set_cached_metrics("bounded-run", rows)

    assert len(stored_rows) == 100
    assert stored_rows[0]["t"] == 150
    assert stored_rows[-1]["t"] == 249
