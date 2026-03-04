#!/bin/bash

conda activate main

# Model 0
echo "--------------------- Running Model 0 scenarios ---------------------"
python scripts/run.py --config abm_results/scenarios/model0/model0_static.yml
python scripts/run.py --config abm_results/scenarios/model0/model0_vol_dex.yml
python scripts/run.py --config abm_results/scenarios/model0/model0_vol_cex.yml
python scripts/run.py --config abm_results/scenarios/model0/model0_tox.yml

# Model 1
echo "--------------------- Running Model 1 scenarios ---------------------"
python scripts/run.py --config abm_results/scenarios/model1/model1_static.yml
python scripts/run.py --config abm_results/scenarios/model1/model1_vol_dex.yml
python scripts/run.py --config abm_results/scenarios/model1/model1_vol_cex.yml
python scripts/run.py --config abm_results/scenarios/model1/model1_tox.yml

# Model 2
echo "--------------------- Running Model 2 scenarios ---------------------"
python scripts/run.py --config abm_results/scenarios/model2/model2_static.yml
python scripts/run.py --config abm_results/scenarios/model2/model2_vol_dex.yml
python scripts/run.py --config abm_results/scenarios/model2/model2_vol_cex.yml
python scripts/run.py --config abm_results/scenarios/model2/model2_tox.yml

# Model 2 with heston theta schedule
echo "--------------------- Running Model 2 with heston theta schedule ---------------------"
python scripts/run.py --config abm_results/scenarios/vol_conditioned_wide.yml