from __future__ import annotations

from pathlib import Path

from abm_webapp.storage import SQLiteLiveSink, read_metrics, read_status


def test_sqlite_live_sink_writes_and_reads(tmp_path: Path) -> None:
    db_path = tmp_path / "live.db"
    sink = SQLiteLiveSink(
        db_path=db_path,
        run_id="unit_test_run",
        params_yaml="fee_mode: static\nsimulate:\n  T: 3\n",
        results_root=tmp_path,
        T=3,
        commit_every=1,
    )
    sink.set_log_path(str(tmp_path / "logs" / "test.txt"))
    sink.record_step(
        dict(
            t=0,
            dex_price=1.0,
            cex_price=1.0,
            cex_sigma=0.0001,
            band_lo=0.99,
            band_hi=1.01,
            sr_pnl_step=0.0,
            noise_pnl_step=0.0,
            arb_pnl_step=0.0,
            lp_pnl_active=0.0,
            lp_pnl_passive=0.0,
            lp_unhedged_active=0.0,
            lp_unhedged_passive=0.0,
            lp_fee_value_total=0.0,
            lp_lvr_total=0.0,
            jiter_pnl=0.0,
            dex_notional_y=0.0,
            d_lvr_total=0.0,
            d_fee_value_total=0.0,
            trader_exec_count=0,
            arb_exec_count=0,
            sr_exec_count=0,
            noise_exec_count=0,
            sr_cex_exec_count=0,
            sr_dex_exec_count=0,
            fee=0.0001,
            fee_sigma=0.0,
            fee_basis_ticks=0.0,
            fee_signal=0.0,
        )
    )
    sink.record_step(
        dict(
            t=1,
            dex_price=1.01,
            cex_price=1.02,
            cex_sigma=0.00012,
            band_lo=1.0,
            band_hi=1.04,
            sr_pnl_step=0.1,
            noise_pnl_step=-0.2,
            arb_pnl_step=0.05,
            lp_pnl_active=0.0,
            lp_pnl_passive=0.0,
            lp_unhedged_active=0.03,
            lp_unhedged_passive=-0.01,
            lp_fee_value_total=0.12,
            lp_lvr_total=0.07,
            jiter_pnl=0.0,
            dex_notional_y=42.0,
            d_lvr_total=0.006,
            d_fee_value_total=0.009,
            trader_exec_count=2,
            arb_exec_count=1,
            sr_exec_count=1,
            noise_exec_count=1,
            sr_cex_exec_count=1,
            sr_dex_exec_count=0,
            fee=0.0001,
            fee_sigma=0.123,
            fee_basis_ticks=5.0,
            fee_signal=-0.4,
        )
    )
    sink.set_status(state="finished", message="ok")
    sink.close()

    status = read_status(db_path)
    assert status is not None
    assert status.run_id == "unit_test_run"
    assert status.state == "finished"
    assert status.t_last == 1
    assert status.log_path.endswith("test.txt")

    rows = read_metrics(db_path)
    assert [r["t"] for r in rows] == [0, 1]
    assert rows[-1]["dex_price"] == 1.01
    assert rows[-1]["noise_pnl_step"] == -0.2
    assert rows[-1]["cex_sigma"] == 0.00012
    assert rows[-1]["lp_unhedged_active"] == 0.03
    assert rows[-1]["lp_lvr_total"] == 0.07
    assert rows[-1]["d_lvr_total"] == 0.006
    assert rows[-1]["sr_exec_count"] == 1
    assert rows[-1]["fee_sigma"] == 0.123
