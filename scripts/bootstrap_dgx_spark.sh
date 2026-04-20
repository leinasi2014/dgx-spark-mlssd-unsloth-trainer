#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${PYTHON_BIN:=python3}"
: "${ML_SSD_GIT_URL:=https://github.com/apple/ml-ssd.git}"
: "${ML_SSD_COMMIT:=main}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[err] python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

ARCH="$($PYTHON_BIN - <<'PY2'
import platform
print(platform.machine())
PY2
)"

if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "[warn] expected ARM64 for DGX Spark, got: $ARCH"
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p third_party
if [[ ! -d third_party/ml-ssd/.git ]]; then
  git clone "$ML_SSD_GIT_URL" third_party/ml-ssd
fi
(
  cd third_party/ml-ssd
  git fetch --all --tags --prune
  git checkout "$ML_SSD_COMMIT"
)

if [[ -f third_party/ml-ssd/pyproject.toml ]]; then
  python -m pip install -e third_party/ml-ssd
fi

python - <<'PY2'
import platform
import shutil
import sys
print('[env] python=', sys.version.split()[0])
print('[env] machine=', platform.machine())
print('[env] free_disk_gb=', round(shutil.disk_usage('.').free / (1024 ** 3), 2))
try:
    import torch
    print('[env] torch=', torch.__version__)
    print('[env] cuda_available=', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('[env] cuda_device_count=', torch.cuda.device_count())
except Exception as exc:
    print('[env] torch_check_failed=', exc)
PY2
