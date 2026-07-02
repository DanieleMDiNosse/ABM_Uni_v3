# Run configuration YAMLs

This directory contains source YAML files used to launch reproducible runs. Generated outputs and run snapshots stay under `abm_results/`.

- `scenarios/`: runnable single-scenario configs for `python -m scripts.run --config ...` and the webapp scenario dropdown.
- `experiments/`: experiment-design configs for `python -m scripts.run_experiment_design --experiment ...`.
- `paper/`: paper figure-runner configs and generated source configs used to reproduce manuscript figures.

Do not move generated `config_snapshot.yml` files here: they are provenance records and should remain next to the output run that produced them.
