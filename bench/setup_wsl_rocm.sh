#!/usr/bin/env bash
# ROCm 7.2 userspace inside WSL2 for an RX 9070 (gfx1201).
# The Windows host already owns the GPU. This script does not install a
# kernel driver. AMD's WSL usecase is:
#   amdgpu-install -y --usecase=wsl,rocm --no-dkms
# Docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html
# The host driver used with this tree is Adrenalin 26.10.1. The 7.2 WSL
# doc names 26.1.1 as the driver it was written against. Do not downgrade
# the host driver from this script.
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
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
sudo usermod -aG render,video "$USER" || true

if ! grep -q HSA_ENABLE_DXG_DETECTION "$HOME/.bashrc" 2>/dev/null; then
  echo 'export HSA_ENABLE_DXG_DETECTION=1' >> "$HOME/.bashrc"
fi

python3 -m venv "$ROOT/.venv-wsl"
# shellcheck disable=SC1091
source "$ROOT/.venv-wsl/bin/activate"
python -m pip install --upgrade pip wheel
# Same torch the Windows ROCm 7.2.1 run used: 2.9.1. The Radeon wheel index
# pulls the matching torchvision and the ROCm Triton build.
python -m pip install torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
python -m pip install numpy pillow

# AMD's WSL note: drop the libhsa copy shipped inside the wheel so the
# userspace ROCm runtime is the one that sees /dev/dxg.
TORCH_LIB="$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")')"
rm -f "$TORCH_LIB"/libhsa-runtime64.so*

python - << 'PY'
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
echo "Next: source .venv-wsl/bin/activate && export PYTHONPATH=src:. && export HSA_ENABLE_DXG_DETECTION=1"
echo "Then: python bench/train_max_v2.py --workers 2 --out results/train_max_v2 --materials data/materials --probe-steps 400"
