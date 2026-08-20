#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 && -z "${SCORE_FOLDER_PATH:-}" ]]; then
  echo "Usage: $0 <episode_dir> [trajectory_index]" >&2
  echo "Or set SCORE_FOLDER_PATH before running." >&2
  exit 2
fi

if [[ $# -ge 1 ]]; then
  export SCORE_FOLDER_PATH="$1"
fi

trajectory_index="${2:-0}"
export ROBOT_NAME="marvin_pro"
export PYTHONUNBUFFERED=1

cd "${SCRIPT_DIR}"
exec python3 replay.py "${trajectory_index}"
