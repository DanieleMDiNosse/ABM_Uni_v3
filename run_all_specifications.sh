#!/usr/bin/env bash
set -euo pipefail

# Conda activation in non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

# Avoid "parallelism × parallelism" (common perf killer)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

# Build a list of ONE logical CPU per PHYSICAL core, split by NUMA node,
# then interleave nodes (0,1,0,1,...) to balance sockets.
mapfile -t NODE0_CPUS < <(
  lscpu -p=CPU,SOCKET,CORE,NODE | grep -v '^#' |
  awk -F, '!seen[$2":"$3]++ && $4==0 {print $1}'
)
mapfile -t NODE1_CPUS < <(
  lscpu -p=CPU,SOCKET,CORE,NODE | grep -v '^#' |
  awk -F, '!seen[$2":"$3]++ && $4==1 {print $1}'
)

PHYS_CPUS=()
for ((i=0; i<${#NODE0_CPUS[@]} || i<${#NODE1_CPUS[@]}; i++)); do
  [[ $i -lt ${#NODE0_CPUS[@]} ]] && PHYS_CPUS+=("${NODE0_CPUS[$i]}")
  [[ $i -lt ${#NODE1_CPUS[@]} ]] && PHYS_CPUS+=("${NODE1_CPUS[$i]}")
done

echo "Physical-core CPU list (one thread/core, NUMA-interleaved): ${PHYS_CPUS[*]}"

configs=(
  abm_results/scenarios/model0/model0_static.yml
  abm_results/scenarios/model0/model0_vol_dex.yml
  abm_results/scenarios/model0/model0_vol_cex.yml
  abm_results/scenarios/model0/model0_tox.yml

  abm_results/scenarios/model1/model1_static.yml
  abm_results/scenarios/model1/model1_vol_dex.yml
  abm_results/scenarios/model1/model1_vol_cex.yml
  abm_results/scenarios/model1/model1_tox.yml

  abm_results/scenarios/model2/model2_static.yml
  abm_results/scenarios/model2/model2_vol_dex.yml
  abm_results/scenarios/model2/model2_vol_cex.yml
  abm_results/scenarios/model2/model2_tox.yml

  abm_results/scenarios/vol_conditioned_wide.yml
)

# Prefer numactl (pins memory local to the CPU) if available; else taskset.
if command -v numactl >/dev/null 2>&1; then
  pin_run() { numactl --physcpubind="$1" --localalloc "${@:2}"; }
else
  pin_run() { taskset -c "$1" "${@:2}"; }
fi

pids=()
i=0
for cfg in "${configs[@]}"; do
  cpu="${PHYS_CPUS[$((i % ${#PHYS_CPUS[@]}))]}"
  log="logs/$(basename "$cfg" .yml).log"
  echo "Launching $cfg on CPU $cpu -> $log"

  pin_run "$cpu" python scripts/run.py --config "$cfg" >"$log" 2>&1 &
  pids+=("$!")
  i=$((i+1))
done

trap 'echo "Caught signal, killing children..."; kill 0' INT TERM

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
exit "$fail"