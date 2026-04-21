#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${PYTHON_BIN:=python3}"
: "${ML_SSD_GIT_URL:=https://github.com/apple/ml-ssd.git}"
: "${ML_SSD_COMMIT:=15c429241729df2704f926bd3e6ac19cf502f245}"
: "${INSTALL_ML_SSD:=0}"
: "${ML_SSD_CHECKOUT_DIR:=third_party/ml-ssd}"

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
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[err] expected virtualenv python at $VENV_PYTHON" >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r requirements.txt

if [[ "$INSTALL_ML_SSD" == "1" ]]; then
  mkdir -p "$(dirname "$ML_SSD_CHECKOUT_DIR")"
  if [[ ! -d "$ML_SSD_CHECKOUT_DIR/.git" ]]; then
    git clone "$ML_SSD_GIT_URL" "$ML_SSD_CHECKOUT_DIR"
  fi
  (
    cd "$ML_SSD_CHECKOUT_DIR"
    git fetch --all --tags --prune
    git checkout "$ML_SSD_COMMIT"
  )

  if [[ -f "$ML_SSD_CHECKOUT_DIR/pyproject.toml" ]]; then
    "$VENV_PYTHON" -m pip install -e "$ML_SSD_CHECKOUT_DIR"
  fi
else
  echo "[info] skipping optional ml-ssd checkout (set INSTALL_ML_SSD=1 to enable)"
fi

"$VENV_PYTHON" - <<'PY2'
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

echo "[next] use $VENV_PYTHON for subsequent commands"
