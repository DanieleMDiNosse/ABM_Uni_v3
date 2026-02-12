#!/usr/bin/env bash
set -euo pipefail

# Run `run_multiple.py` for static fees under three LP/JIT configurations.
#
# Usage:
#   ./run_multiple_static_sims.sh [run_multiple.py args...]
#
# Examples:
#   ./run_multiple_static_sims.sh --runs 20 --seed-base 1 --max-workers 8
#
# Notes:
# - All parameters are inherited from `abm_results/scenarios/test.yml`, except:
#     - `fee_mode` is forced to `static` (via `--fee-modes static`)
#     - `passive_lp_share` and `p_jit` are overridden per model
# - Outputs are written under:
#     `abm_results/scenarios/<modelN>/multi_runs/{png,html}/`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_CONFIG="abm_results/scenarios/test.yml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 1
fi

# Activate the main conda environment when available (recommended for this repo).
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate main
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

make_config() {
  local name="$1"
  local passive_lp_share="$2"
  local p_jit="$3"
  local out_path="${TMP_DIR}/${name}.yml"

  python - "${BASE_CONFIG}" "${out_path}" "${name}" "${passive_lp_share}" "${p_jit}" <<'PY'
import re
import sys
from pathlib import Path


def _replace_required_line(text: str, *, pattern: str, replacement: str) -> str:
    """Replace exactly one required YAML line, failing if missing."""
    new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if n == 0:
        raise SystemExit(f"Expected to find line matching pattern: {pattern!r}")
    return new_text


def main() -> None:
    src, dst, name, passive_lp_share, p_jit = sys.argv[1:]
    text = Path(src).read_text(encoding="utf-8")

    # Keep base config but make the intent explicit.
    text = _replace_required_line(text, pattern=r"^fee_mode:.*$", replacement="fee_mode: static")

    # Ensure labels/logs reflect the model name.
    text = _replace_required_line(
        text, pattern=r"^  config_name:.*$", replacement=f"  config_name: {name}"
    )

    # Override the two requested knobs.
    text = _replace_required_line(
        text,
        pattern=r"^  passive_lp_share:.*$",
        replacement=f"  passive_lp_share: {passive_lp_share}",
    )
    text = _replace_required_line(text, pattern=r"^  p_jit:.*$", replacement=f"  p_jit: {p_jit}")

    Path(dst).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
PY

  echo "${out_path}"
}

cfg1="$(make_config model0 1 0)"
cfg2="$(make_config model1 0.5 0)"
cfg3="$(make_config model2 0.5 1)"

echo [static] model0: passive_lp_share=1, p_jit=0
python run_multiple.py --config "${cfg1}" --fee-modes static "$@"

echo [static] model1: passive_lp_share=0.5, p_jit=0
python run_multiple.py --config "${cfg2}" --fee-modes static "$@"

echo [static] model2: passive_lp_share=0.5, p_jit=1
python run_multiple.py --config "${cfg3}" --fee-modes static "$@"
