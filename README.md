# ABM_Uni_v3

Agent-Based Model (ABM) simulator for a Uniswap v3 style pool (simulation engine + optional Dash webapp).

## Quickstart

```bash
conda env create -f environment.yml
conda activate main
python -m scripts.run --config configs/scenarios/section4_microstructure_model0_static.yml
```

Run config YAMLs live under `configs/`; outputs are written under `abm_results/scenarios/<scenario_name>/`.

## Tests

```bash
conda activate main
pytest -q
```

## Docs / Paper / Webapp

- Docs site sources live in `docs/` (see `docs/index.md`).
- Manuscript + paper-stable figures live in `paper/` (see `paper/AGENTS.md`).
- Webapp details live in `abm_webapp/README.md` (`python -m abm_webapp` or `amm-abm-web` when installed).
