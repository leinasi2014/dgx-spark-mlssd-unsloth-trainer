#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

echo "agent_evals bootstrap complete."
echo "Next:"
echo "  .venv/bin/agent-evals doctor"
echo "  .venv/bin/agent-evals fetch --benchmark all"
