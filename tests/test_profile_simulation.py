from __future__ import annotations

import cProfile

import pytest

from scripts.profile_simulation import (
    _build_report,
    _estimate_result_payload,
    _top_profile_lines,
    parse_args,
)


def _tiny_profile() -> cProfile.Profile:
    profiler = cProfile.Profile()

    def hot_loop() -> int:
        total = 0
        for i in range(20):
            total += i * i
        return total

    profiler.enable()
    hot_loop()
    profiler.disable()
    return profiler


def test_parse_args_accepts_explicit_light_mode() -> None:
    args = parse_args(["--T", "25", "--light-mode", "--top-n", "3"])

    assert args.T == 25
    assert args.light_mode is True
    assert args.full_mode is False
    assert args.top_n == 3


def test_parse_args_rejects_non_positive_horizon() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--T", "0"])


def test_estimate_result_payload_counts_lists_and_array_like_sizes() -> None:
    class FakeArray:
        size = 11

    keys, elements = _estimate_result_payload(
        {
            "prices": [1.0, 2.0, 3.0],
            "micro": (4.0, 5.0),
            "array_like": FakeArray(),
            "nested": {"a": 1, "b": 2},
            "label": "static",
        }
    )

    assert keys == 5
    assert elements == 3 + 2 + 11 + 2


def test_build_report_uses_dynamic_hotspots() -> None:
    profiler = _tiny_profile()
    hot_lines = _top_profile_lines(profiler, sort_key="tottime", top_n=3)

    report = _build_report(
        T=20,
        block_time=5,
        light_mode=True,
        wall_seconds=1.25,
        rss_peak_kib=2048,
        tracemalloc_peak=10_000,
        tracemalloc_current=5_000,
        return_dict_keys=4,
        return_dict_elements=100,
        disk_bytes=2048,
        top_memory_lines=["  10.0 KiB      10 blocks  fake.py:1"],
        top_cumulative_lines=hot_lines,
        top_tottime_lines=hot_lines,
        extrapolate_to=100,
    )

    assert "--- HOT FUNCTIONS (cumulative time) ---" in report
    assert "--- HOT FUNCTIONS (self time) ---" in report
    assert "--- SCALING NOTES ---" in report
    assert "This run used light mode" in report
    assert "lp_token0_exposure" not in report
