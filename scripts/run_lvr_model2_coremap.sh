#!/usr/bin/env bash
set -euo pipefail

# Run LVR_vs_blocksize for all model2 scenarios plus vol_conditioned_wide,
# binding each scenario run to a CPU core (scenario -> core mapping).
#
# Usage:
#   bash scripts/run_lvr_model2_coremap.sh --runs 50 --fee-definition flow
#
# Optional environment variable:
#   CORE_IDS="0,1,2,3,4"   # explicit cores to use (default: all cores from nproc)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

_activate_conda_main() {
    if ! command -v conda >/dev/null 2>&1; then
        if [[ -n "${CONDA_EXE:-}" ]]; then
            # shellcheck disable=SC1090
            source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
        elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        else
            echo "[error] conda is not available in this shell." >&2
            exit 1
        fi
    else
        eval "$(conda shell.bash hook)"
    fi

    conda activate main
}

_parse_core_ids() {
    local -n _out=$1
    _out=()

    if [[ -n "${CORE_IDS:-}" ]]; then
        IFS=',' read -r -a _out <<< "${CORE_IDS}"
        return
    fi

    local ncores
    ncores="$(nproc)"
    if [[ "${ncores}" -le 0 ]]; then
        echo "[error] could not detect CPU cores via nproc." >&2
        exit 1
    fi
    for ((i = 0; i < ncores; i++)); do
        _out+=("${i}")
    done
}

_has_max_workers_flag() {
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "--max-workers" || "${arg}" == --max-workers=* ]]; then
            return 0
        fi
    done
    return 1
}

main() {
    if ! command -v taskset >/dev/null 2>&1; then
        echo "[error] taskset is required but not found." >&2
        exit 1
    fi

    cd "${REPO_ROOT}"
    _activate_conda_main

    local -a scenarios=()
    mapfile -t scenarios < <(find "${REPO_ROOT}/abm_results/scenarios/model2" \
        -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) | sort)

    scenarios+=("${REPO_ROOT}/abm_results/scenarios/vol_conditioned_wide.yml")

    if [[ "${#scenarios[@]}" -eq 0 ]]; then
        echo "[error] no scenarios found." >&2
        exit 1
    fi

    local s
    for s in "${scenarios[@]}"; do
        if [[ ! -f "${s}" ]]; then
            echo "[error] missing scenario file: ${s}" >&2
            exit 1
        fi
    done

    local -a core_ids=()
    _parse_core_ids core_ids
    if [[ "${#core_ids[@]}" -eq 0 ]]; then
        echo "[error] empty core list." >&2
        exit 1
    fi

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_dir="${REPO_ROOT}/abm_results/scenarios/_batch_logs/lvr_vs_blocksize_model2_${ts}"
    mkdir -p "${log_dir}"

    local -a extra_args=("$@")
    local -a base_cmd=(python -m scripts.LVR_vs_blocksize)
    if ! _has_max_workers_flag "${extra_args[@]}"; then
        # One simulation process per mapped core by default.
        base_cmd+=(--max-workers 1)
    fi

    echo "[info] logs: ${log_dir}"
    echo "[info] scenarios: ${#scenarios[@]}"
    echo "[info] cores: ${core_ids[*]}"

    local -a pids=()
    local idx core scenario stem log_file pid
    for idx in "${!scenarios[@]}"; do
        scenario="${scenarios[$idx]}"
        core="${core_ids[$((idx % ${#core_ids[@]}))]}"
        stem="$(basename "${scenario}")"
        stem="${stem%.yml}"
        stem="${stem%.yaml}"
        log_file="${log_dir}/${idx}_${stem}.log"

        echo "[map] ${scenario} -> core ${core}"
        taskset -c "${core}" "${base_cmd[@]}" --config "${scenario}" "${extra_args[@]}" >"${log_file}" 2>&1 &
        pid=$!
        pids+=("${pid}")
    done

    local rc=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            rc=1
        fi
    done

    if [[ "${rc}" -ne 0 ]]; then
        echo "[error] one or more scenario runs failed. Check logs in ${log_dir}" >&2
        exit "${rc}"
    fi

    echo "[ok] all scenario runs completed. Logs in ${log_dir}"
}

main "$@"
