from pathlib import Path

import yaml

from scripts.analysis import run_paper_figures


def _write_base_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "simulate": {
                    "config_name": "base_model0",
                    "fee_mode": "toxicity",
                    "fee_use_ewma": True,
                    "passive_lp_share": 1.0,
                    "p_jit": 0.0,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_scenario_params_can_disable_fee_ewma_for_cross_scenario_runs() -> None:
    params = run_paper_figures._scenario_params(
        {"fee_use_ewma": True, "other": 1},
        "Model2",
        "toxicity",
        fee_use_ewma=False,
    )

    assert params["fee_mode"] == "toxicity"
    assert params["passive_lp_share"] == 0.5
    assert params["p_jit"] == 1
    assert params["fee_use_ewma"] is False


def test_model2_blocksize_config_can_disable_fee_ewma(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    _write_base_config(base_config)

    generated = run_paper_figures._write_model2_blocksize_config(
        base_config,
        "linear_asymmetric",
        tmp_path / "generated",
        fee_use_ewma=False,
    )

    data = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert data["fee_mode"] == "linear_asymmetric"
    assert data["simulate"]["config_name"] == "model2_linear_asym"
    assert data["simulate"]["fee_mode"] == "linear_asymmetric"
    assert data["simulate"]["passive_lp_share"] == 0.5
    assert data["simulate"]["p_jit"] == 1
    assert data["simulate"]["fee_use_ewma"] is False


def test_fee_override_config_can_disable_ewma_for_microstructure_diagnostics(tmp_path: Path) -> None:
    base_config = tmp_path / "toxicity.yml"
    _write_base_config(base_config)

    generated = run_paper_figures._write_fee_override_config(
        base_config,
        tmp_path / "generated" / "toxicity_no_ewma.yml",
        fee_use_ewma=False,
    )

    data = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert data["simulate"]["fee_mode"] == "toxicity"
    assert data["simulate"]["fee_use_ewma"] is False
