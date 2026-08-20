#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${MARVIN_ENV_NAME:-marvin-physical-validation}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

if [[ -z "${CONDA_BIN}" ]]; then
  echo "Conda was not found. Install Miniconda/Miniforge or add conda to PATH." >&2
  exit 2
fi

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import sys" >/dev/null 2>&1; then
  echo "[ENV] Creating ${ENV_NAME} with Python 3.10"
  "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.10 pip -y
else
  echo "[ENV] Reusing existing environment: ${ENV_NAME}"
fi

echo "[ENV] Installing project dependencies"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "[ENV] Running Marvin model smoke test"
cd "${SCRIPT_DIR}"
MUJOCO_GL=egl "${CONDA_BIN}" run -n "${ENV_NAME}" python smoke_test.py

echo
echo "Environment is ready. Activate it with:"
echo "  conda activate ${ENV_NAME}"
