#!/usr/bin/env bash
# ROCm 7.2 userspace inside WSL2 for an RX 9070 (gfx1201).
# The Windows host already owns the GPU. This script does not install a
# kernel driver and does not change the host Adrenalin driver (26.10.1 here).
#
# ROCm 7.2 rejects `amdgpu-install --usecase=wsl` with
# "Usecase implementation 'wsl' is not supported or invalid".
# Install the ROCm userspace, then the librocdxg runtime:
#   sudo amdgpu-install -y --usecase=rocm --no-dkms
#   rocdxg-roct_1.2.2_amd64.deb
#   https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
# Legacy ROCm (this 7.2 stack, before 7.13) needs HSA_ENABLE_DXG_DETECTION=1
# so libhsa loads librocdxg and sees /dev/dxg.
# The torch wheel ships its own libhsa-runtime64.so. That copy does not load
# librocdxg. Remove it after pip so the system runtime is the one that is used.
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "This script is for Ubuntu under WSL2. /proc/version does not mention Microsoft."
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DEBIAN_FRONTEND=noninteractive
export HSA_ENABLE_DXG_DETECTION=1

sudo apt-get update
sudo apt-get install -y wget ca-certificates python3 python3-venv python3-pip python3-dev build-essential

DEB="amdgpu-install_7.2.70200-1_all.deb"
URL="https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/${DEB}"
if ! dpkg -s amdgpu-install >/dev/null 2>&1; then
  wget -O "/tmp/${DEB}" "$URL"
  sudo apt-get install -y "/tmp/${DEB}"
fi
sudo amdgpu-install -y --usecase=rocm --no-dkms
sudo ldconfig

ROCDXG_DEB="rocdxg-roct_1.2.2_amd64.deb"
ROCDXG_URL="https://github.com/ROCm/librocdxg/releases/download/v1.2.2/${ROCDXG_DEB}"
# Digest published on the v1.2.2 release asset.
ROCDXG_SHA256="28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541"
wget -O "/tmp/${ROCDXG_DEB}" "$ROCDXG_URL"
printf '%s  %s\n' "$ROCDXG_SHA256" "/tmp/${ROCDXG_DEB}" | sha256sum -c -
sudo apt-get install -y "/tmp/${ROCDXG_DEB}" || {
  sudo dpkg -i "/tmp/${ROCDXG_DEB}"
  sudo apt-get install -y -f
}
sudo ldconfig
# /opt/rocm is a symlink. find does not descend into a symlink unless -L
# is set (a trailing slash on the start path has the same effect).
if ! find -L /opt/rocm/ /usr/lib /usr/local/lib -name 'librocdxg.so*' -print -quit 2>/dev/null | grep -q .; then
  echo "librocdxg.so was not found after installing ${ROCDXG_DEB}"
  exit 1
fi

sudo usermod -aG render,video "$USER" || true

if ! grep -q HSA_ENABLE_DXG_DETECTION "$HOME/.bashrc" 2>/dev/null; then
  echo 'export HSA_ENABLE_DXG_DETECTION=1' >> "$HOME/.bashrc"
fi

python3 -m venv "$ROOT/.venv-wsl"
if ! grep -q HSA_ENABLE_DXG_DETECTION "$ROOT/.venv-wsl/bin/activate" 2>/dev/null; then
  echo 'export HSA_ENABLE_DXG_DETECTION=1' >> "$ROOT/.venv-wsl/bin/activate"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-wsl/bin/activate"
# libhsa reads this when torch loads it. Set it again here so a later
# import cannot run with the variable unset.
export HSA_ENABLE_DXG_DETECTION=1
python -m pip install --upgrade pip wheel
python -m pip install numpy pillow
# torch 2.9.1+rocm6.4 from the rocm6.4 index. The repo.radeon.com manylinux
# directory is not a pip index. This build sees an RX 9070 when the system
# ROCm 7.2 libhsa and librocdxg are the ones that load.
# Install from the rocm6.4 index, not from bare wheel URLs. A direct torch
# wheel is 4.4 GB and then pip looks up pytorch-triton-rocm==3.5.1 on PyPI,
# where it does not exist. The index hosts torch and pytorch-triton-rocm
# together, and it resolves that pin before downloading the wheel. Skip the
# download when this venv already has torch 2.9.1+rocm.
export HSA_ENABLE_DXG_DETECTION=1
if python -c 'import os; os.environ["HSA_ENABLE_DXG_DETECTION"]="1"; import torch; raise SystemExit(0 if str(torch.__version__).startswith("2.9.1") and "rocm" in str(torch.__version__) else 1)'; then
  echo "torch $(python -c 'import os; os.environ["HSA_ENABLE_DXG_DETECTION"]="1"; import torch; print(torch.__version__)') already installed"
else
  set +e
  python -m pip install --index-url https://download.pytorch.org/whl/rocm6.4 \
    "torch==2.9.1" "torchvision==0.24.0" "torchaudio==2.9.0"
  pip_status=$?
  set -e
  if [[ "$pip_status" -ne 0 ]]; then
    echo "pip install of torch 2.9.1+rocm6.4 failed with status ${pip_status}" >&2
    exit "$pip_status"
  fi
fi

# Drop every libhsa-runtime64 the wheel unpacked. The dynamic linker then
# uses the system libhsa, which is the one that loads librocdxg.
export HSA_ENABLE_DXG_DETECTION=1
TORCH_LIB="$(python -c 'import os; os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"; import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")')"
mapfile -t HSA_BUNDLED < <(find "$TORCH_LIB" -name 'libhsa-runtime64.so*' -print)
if ((${#HSA_BUNDLED[@]})); then
  rm -f "${HSA_BUNDLED[@]}"
  printf 'removed bundled libhsa-runtime64:\n'
  printf '  %s\n' "${HSA_BUNDLED[@]}"
fi
mapfile -t HSA_LEFT < <(find "$TORCH_LIB" -name 'libhsa-runtime64.so*' -print)
if ((${#HSA_LEFT[@]})); then
  echo "torch still contains libhsa-runtime64; the system runtime would not be used"
  printf '  %s\n' "${HSA_LEFT[@]}"
  exit 1
fi
echo "system libhsa-runtime64:"
ldconfig -p | grep 'libhsa-runtime64' || echo "warning: ldconfig has no libhsa-runtime64"

export HSA_ENABLE_DXG_DETECTION=1
python - << 'PY'
import os
os.environ["HSA_ENABLE_DXG_DETECTION"] = "1"
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false. Check HSA_ENABLE_DXG_DETECTION and rocminfo.")
name = torch.cuda.get_device_name(0)
print("device", name)
if "9070" not in name and "gfx1201" not in name.lower():
    print("warning: expected RX 9070 / gfx1201, got", name)
# A one-op compile. Failure is printed and does not fail the install.
try:
    fn = torch.compile(lambda x: x * 2 + 1, mode="reduce-overhead")
    y = fn(torch.ones(8, device="cuda"))
    torch.cuda.synchronize()
    print("compile_reduce_overhead", float(y[0]))
except Exception as exc:
    print("compile_reduce_overhead_failed", exc)
PY

echo "venv: $ROOT/.venv-wsl"
echo "If this user was just added to render and video, open a new shell before training."
echo "Next: source .venv-wsl/bin/activate"
echo "Then: export HSA_ENABLE_DXG_DETECTION=1"
echo "      export PYTHONPATH=src:."
echo "      python bench/train_max_v2.py --workers 3 --out results/train_max_v2 --materials data/materials --probe-steps 400"
echo "Set HSA_ENABLE_DXG_DETECTION before Python starts. torch.cuda.is_available() stays false if the variable is set after import torch."
