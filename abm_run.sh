#!/bin/bash
eval "$(/cluster/shared/software/miniconda3/bin/conda shell.bash hook)"
conda activate abm_uni_v3
python run_parameter_surface_nd_pnl_fee_dashboard.py --recompute
