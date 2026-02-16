import pytest

from scripts.run_multiple import _resolve_pnl_specs


def _pnl_keys(params):
    return {spec.key for spec in _resolve_pnl_specs(params)}


def test_resolve_pnl_specs_accepts_lp_passive_share_alias_and_hides_active():
    keys = _pnl_keys({"lp_passive_share": 1.0})
    assert "lp_pnl_active" not in keys
    assert "lp_pnl_passive" in keys


@pytest.mark.parametrize(
    "smart_trades_per_block, noise_trades_per_block, expected_absent",
    [
        (0.0, 1.0, {"smart_router_pnl_cum"}),
        (1.0, 0.0, {"noise_trader_pnl_cum"}),
        (0.0, 0.0, {"smart_router_pnl_cum", "noise_trader_pnl_cum"}),
    ],
)
def test_resolve_pnl_specs_skips_inactive_traders(
    smart_trades_per_block: float,
    noise_trades_per_block: float,
    expected_absent: set[str],
):
    params = {
        "passive_lp_share": 0.5,
        "smart_trades_per_block": smart_trades_per_block,
        "noise_trades_per_block": noise_trades_per_block,
    }
    keys = _pnl_keys(params)
    assert expected_absent.isdisjoint(keys)
    # Ensure LP and arb cohorts still show up as long as they are configured.
    assert "lp_pnl_active" in keys
    assert "lp_pnl_passive" in keys
    assert "arb_pnl_cum" in keys
