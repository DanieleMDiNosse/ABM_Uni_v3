#!/usr/bin/env bash
set -euo pipefail

# Run LVR_vs_blocksize for all model2 scenarios plus vol_conditioned_wide,
# binding each scenario run to a dedicated CPU set.
#
# Usage:
#   bash scripts/run_lvr_model2_coremap.sh --runs 50 --fee-definition flow
#   bash scripts/run_lvr_model2_coremap.sh --runs 50 --fee-definition flow \
#     --max-workers 10 --scenario-slots 1
#
# Optional environment variable:
#   CORE_IDS="0,1,2,3,4"   # explicit cores to use (default: all cores from nproc)
#
# Parallelization procedure:
#   1. Build the available CPU list from CORE_IDS or from `nproc`.
#   2. Read `--max-workers` from the CLI. If it is omitted, this wrapper forces
#      `scripts.LVR_vs_blocksize` to run with `--max-workers 1`.
#   3. Reserve a non-overlapping CPU group of size `max_workers` for each
#      concurrent scenario. For example, with 16 cores and `--max-workers 4`,
#      the wrapper creates 4 groups: `0,1,2,3`, `4,5,6,7`, `8,9,10,11`,
#      `12,13,14,15`.
#   4. Launch at most `floor(n_cores / max_workers)` scenarios at a time, or
#      fewer if `--scenario-slots` is provided. Each scenario is pinned via
#      `taskset` to one of those CPU groups. `scripts.LVR_vs_blocksize` can then
#      use its internal `ProcessPoolExecutor` on that CPU group.
#   5. Wait for the whole batch to finish before launching the next batch.
#
# Notes:
#   - If `n_cores` is not divisible by `max_workers`, the remainder cores are
#     left idle so CPU groups stay disjoint.
#   - If `--max-workers` exceeds the number of available cores, the wrapper
#     exits with an error rather than silently oversubscribing.
#   - `--scenario-slots` is a wrapper-only argument. It caps how many scenario
#     YAMLs are launched concurrently without changing the inner
#     `scripts.LVR_vs_blocksize --max-workers` setting.

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
        local -a raw_core_ids=()
        local core
        IFS=',' read -r -a raw_core_ids <<< "${CORE_IDS}"
        for core in "${raw_core_ids[@]}"; do
            core="${core//[[:space:]]/}"
            if [[ -z "${core}" ]]; then
                echo "[error] CORE_IDS contains an empty entry." >&2
                exit 1
            fi
            if [[ ! "${core}" =~ ^[0-9]+$ ]]; then
                echo "[error] CORE_IDS entries must be non-negative integers. Got: ${core}" >&2
                exit 1
            fi
            _out+=("${core}")
        done
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

_split_wrapper_args() {
    local -n _forwarded_args=$1
    local -n _max_workers_out=$2
    local -n _scenario_slots_out=$3
    shift 3

    _forwarded_args=()
    _max_workers_out=""
    _scenario_slots_out=""

    local arg
    local expect_max_workers=0
    local expect_scenario_slots=0
    for arg in "$@"; do
        if [[ "${expect_max_workers}" -eq 1 ]]; then
            _max_workers_out="${arg}"
            _forwarded_args+=("${arg}")
            expect_max_workers=0
            continue
        fi
        if [[ "${expect_scenario_slots}" -eq 1 ]]; then
            _scenario_slots_out="${arg}"
            expect_scenario_slots=0
            continue
        fi

        case "${arg}" in
            --max-workers)
                _forwarded_args+=("${arg}")
                expect_max_workers=1
                ;;
            --max-workers=*)
                _max_workers_out="${arg#--max-workers=}"
                _forwarded_args+=("${arg}")
                ;;
            --scenario-slots)
                expect_scenario_slots=1
                ;;
            --scenario-slots=*)
                _scenario_slots_out="${arg#--scenario-slots=}"
                ;;
            *)
                _forwarded_args+=("${arg}")
                ;;
        esac
    done

    if [[ "${expect_max_workers}" -eq 1 ]]; then
        echo "[error] --max-workers requires a value." >&2
        exit 1
    fi
    if [[ "${expect_scenario_slots}" -eq 1 ]]; then
        echo "[error] --scenario-slots requires a value." >&2
        exit 1
    fi
}

_require_positive_int() {
    local value=$1
    local label=$2

    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[error] ${label} must be a positive integer. Got: ${value}" >&2
        exit 1
    fi
}

_build_core_groups() {
    local -n _out=$1
    local workers_per_scenario=$2
    shift 2

    local -a cores=("$@")
    local ncores=${#cores[@]}

    _out=()

    _require_positive_int "${workers_per_scenario}" "--max-workers"

    if [[ "${ncores}" -lt "${workers_per_scenario}" ]]; then
        echo "[error] requested --max-workers=${workers_per_scenario} but only ${ncores} cores are available." >&2
        exit 1
    fi

    local ngroups=$((ncores / workers_per_scenario))
    local g idx start end group
    for ((g = 0; g < ngroups; g++)); do
        start=$((g * workers_per_scenario))
        end=$((start + workers_per_scenario))
        group=""
        for ((idx = start; idx < end; idx++)); do
            if [[ -n "${group}" ]]; then
                group+=","
            fi
            group+="${cores[$idx]}"
        done
        _out+=("${group}")
    done
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

    local -a forwarded_args=()
    local -a base_cmd=(python -m scripts.LVR_vs_blocksize)
    local max_workers_arg=""
    local scenario_slots_arg=""
    _split_wrapper_args forwarded_args max_workers_arg scenario_slots_arg "$@"

    local workers_per_scenario=1
    if [[ -n "${max_workers_arg}" ]]; then
        _require_positive_int "${max_workers_arg}" "--max-workers"
        workers_per_scenario="${max_workers_arg}"
    else
        # Keep the default conservative: one worker per scenario unless the
        # user explicitly asks for internal parallelism.
        base_cmd+=(--max-workers 1)
    fi

    local -a core_groups=()
    _build_core_groups core_groups "${workers_per_scenario}" "${core_ids[@]}"

    local max_concurrent_slots=${#core_groups[@]}
    local concurrent_slots=${max_concurrent_slots}
    if [[ -n "${scenario_slots_arg}" ]]; then
        _require_positive_int "${scenario_slots_arg}" "--scenario-slots"
        if [[ "${scenario_slots_arg}" -gt "${max_concurrent_slots}" ]]; then
            echo "[error] requested --scenario-slots=${scenario_slots_arg} but only ${max_concurrent_slots} disjoint scenario slots are available." >&2
            exit 1
        fi
        concurrent_slots=${scenario_slots_arg}
    fi

    local total_scenarios=${#scenarios[@]}
    local total_batches=$(((total_scenarios + concurrent_slots - 1) / concurrent_slots))
    local idle_cores_disjoint=$(( ${#core_ids[@]} % workers_per_scenario ))
    local idle_core_groups=$(( max_concurrent_slots - concurrent_slots ))
    local idle_cores_slot_cap=$(( idle_core_groups * workers_per_scenario ))

    echo "[info] logs: ${log_dir}"
    echo "[info] scenarios: ${total_scenarios}"
    echo "[info] cores: ${core_ids[*]}"
    echo "[info] workers per scenario: ${workers_per_scenario}"
    if [[ -n "${scenario_slots_arg}" ]]; then
        echo "[info] requested scenario slots: ${scenario_slots_arg}"
    else
        echo "[info] requested scenario slots: auto"
    fi
    echo "[info] max disjoint scenario slots from cores: ${max_concurrent_slots}"
    echo "[info] concurrent scenario slots: ${concurrent_slots}"
    echo "[info] total batches: ${total_batches}"
    if [[ "${idle_cores_disjoint}" -gt 0 ]]; then
        echo "[info] idle cores per batch from disjoint core groups: ${idle_cores_disjoint}"
    fi
    if [[ "${idle_cores_slot_cap}" -gt 0 ]]; then
        echo "[info] idle cores per batch from --scenario-slots cap: ${idle_cores_slot_cap}"
    fi

    local rc=0
    local batch_idx slot idx scenario stem log_file pid core_group
    local -a pids=()
    for ((batch_idx = 0; batch_idx < total_batches; batch_idx++)); do
        echo "[batch $((batch_idx + 1))/${total_batches}] launching"
        pids=()
        for ((slot = 0; slot < concurrent_slots; slot++)); do
            idx=$((batch_idx * concurrent_slots + slot))
            if [[ "${idx}" -ge "${total_scenarios}" ]]; then
                break
            fi

            scenario="${scenarios[$idx]}"
            core_group="${core_groups[$slot]}"
            stem="$(basename "${scenario}")"
            stem="${stem%.yml}"
            stem="${stem%.yaml}"
            log_file="${log_dir}/${idx}_${stem}.log"

            echo "[map] batch $((batch_idx + 1))/${total_batches} | ${scenario} -> cores ${core_group}"
            taskset -c "${core_group}" "${base_cmd[@]}" --config "${scenario}" "${forwarded_args[@]}" >"${log_file}" 2>&1 &
            pid=$!
            pids+=("${pid}")
        done

        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                rc=1
            fi
        done
    done

    if [[ "${rc}" -ne 0 ]]; then
        echo "[error] one or more scenario runs failed. Check logs in ${log_dir}" >&2
        exit "${rc}"
    fi

    echo "[ok] all scenario runs completed. Logs in ${log_dir}"
}

main "$@"
