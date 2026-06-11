#!/bin/bash
#SBATCH --job-name=install_unified
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/install.%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/install.%j.err

# =============================================================
# 02_install_unified.sh
# Builds ONE shared virtualenv for all 5 models.
# Pins basicsr==1.4.2 to match the group's unified environment.
# Submit with: sbatch 02_install_unified.sh
# =============================================================
# Dependencies verified against:
#   Uformer   : requirements.txt → einops, timm, skimage, warmup-scheduler
#   KAIR      : requirements.txt → scipy, opencv, tqdm
#   Restormer : requirements.txt → einops, pyyaml
#   NAFNet    : requirements.txt → basicsr, einops, lmdb
#   HAT       : requirements.txt → basicsr, einops, facexlib
# =============================================================

set -euo pipefail

VENV_DIR="$VSC_DATA/projects/unified/venv"
CODE_DIR="$VSC_DATA/projects/unified/code"

# ── Modules ─────────────────────────────────────────────────
module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "System Python:"
which python && python -V

# ── Create fresh venv ────────────────────────────────────────
# --system-site-packages reuses cluster's torch, numpy, scipy — do NOT reinstall them
rm -rf "$VENV_DIR"
python -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

echo "Venv Python:"
which python && python -V

python -m pip install --upgrade pip --no-cache-dir

# ── BasicSR 1.4.2 (pinned — group's unified version) ────────
# Install without pulling in a new torch (already provided by cluster modules)
python -m pip install "basicsr==1.4.2" --no-cache-dir

# ── Core shared deps (all 5 models need these) ──────────────
python -m pip install \
    einops \
    timm \
    scikit-image \
    opencv-contrib-python \
    tqdm \
    tensorboard \
    matplotlib \
    Pillow \
    pyyaml \
    lmdb \
    --no-cache-dir

# ── Uformer-specific ─────────────────────────────────────────
# warmup-scheduler: cosine LR warmup used by Uformer training
# ptflops: FLOP counter (optional, non-critical)
python -m pip install warmup-scheduler --no-cache-dir || \
    echo "WARNING: warmup-scheduler install failed — non-critical, continuing."
python -m pip install ptflops natsort --no-cache-dir

# ── KAIR-specific ────────────────────────────────────────────
# KAIR uses its own bundled basicsr-like code.
# No extra pip installs needed beyond the shared deps above.
echo "KAIR: no extra pip installs required beyond shared deps."

# ── NAFNet: install as editable package ─────────────────────
# NAFNet ships its own modified basicsr inside the repo.
# We install it in editable mode so its local basicsr overrides work.
if [ -d "$CODE_DIR/NAFNet" ]; then
    cd "$CODE_DIR/NAFNet"
    python setup.py develop --no_cuda_ext 2>&1 | tail -5
    cd -
else
    echo "WARNING: NAFNet not cloned yet. Run 01_clone_repos.sh first."
fi

# ── HAT: install as editable package ────────────────────────
if [ -d "$CODE_DIR/HAT" ]; then
    cd "$CODE_DIR/HAT"
    python setup.py develop 2>&1 | tail -5
    cd -
else
    echo "WARNING: HAT not cloned yet. Run 01_clone_repos.sh first."
fi

# ── Sanity checks ────────────────────────────────────────────
echo ""
echo "========== Sanity Checks =========="
python -c "import torch;        print('torch       :', torch.__version__)"
python -c "import torch;        print('CUDA        :', torch.cuda.is_available())"
python -c "import numpy;        print('numpy       :', numpy.__version__)"
python -c "import cv2;          print('cv2         : OK')"
python -c "import einops;       print('einops      : OK')"
python -c "import timm;         print('timm        :', timm.__version__)"
python -c "import skimage;      print('scikit-image:', skimage.__version__)"
python -c "import tensorboard;  print('tensorboard : OK')"
python -c "import basicsr;      print('basicsr     :', basicsr.__version__)"
echo "==================================="
echo ""
echo "Venv ready at: $VENV_DIR"
echo ""
echo "IMPORTANT: basicsr version is $(python -c 'import basicsr; print(basicsr.__version__)')"
echo "This must match 1.4.2 — the group's unified environment."
echo ""
echo "Next step: upload datasets, then bash 03_prepare_datasets.sh"

deactivate
