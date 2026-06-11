#!/bin/bash
# =============================================================
# 01_clone_repos.sh
# Run ONCE on the login node after 00_setup_project.sh.
# Clones all 5 model repos from the team's private GitHub org.
# Usage: bash 01_clone_repos.sh
# =============================================================
# Repos: InViLabVirtualStainingBenchmark/{Uformer,KAIR,Restormer,NAFNet,HAT}
# NOTE: These are PRIVATE repos. You must be authenticated on the
# cluster via SSH key or personal access token before running this.
# To check: git clone https://github.com/InViLabVirtualStainingBenchmark/Restormer.git
# If it asks for a password, set up a GitHub personal access token first.
# =============================================================

CODE_DIR="$VSC_DATA/projects/unified/code"

echo "Cloning model repositories into: $CODE_DIR"
mkdir -p "$CODE_DIR"
cd "$CODE_DIR"

ORG="https://github.com/InViLabVirtualStainingBenchmark"

# ── 1. Uformer ──────────────────────────────────────────────
if [ ! -d "Uformer" ]; then
    echo "[1/5] Cloning Uformer..."
    git clone "$ORG/Uformer.git"
else
    echo "[1/5] Uformer already cloned — skipping."
fi

# ── 2. KAIR (SwinIR training framework) ─────────────────────
if [ ! -d "KAIR" ]; then
    echo "[2/5] Cloning KAIR..."
    git clone "$ORG/KAIR.git"
else
    echo "[2/5] KAIR already cloned — skipping."
fi

# ── 3. Restormer ────────────────────────────────────────────
if [ ! -d "Restormer" ]; then
    echo "[3/5] Cloning Restormer..."
    git clone "$ORG/Restormer.git"
else
    echo "[3/5] Restormer already cloned — skipping."
fi

# ── 4. NAFNet ───────────────────────────────────────────────
if [ ! -d "NAFNet" ]; then
    echo "[4/5] Cloning NAFNet..."
    git clone "$ORG/NAFNet.git"
else
    echo "[4/5] NAFNet already cloned — skipping."
fi

# ── 5. HAT ──────────────────────────────────────────────────
if [ ! -d "HAT" ]; then
    echo "[5/5] Cloning HAT..."
    git clone "$ORG/HAT.git"
else
    echo "[5/5] HAT already cloned — skipping."
fi

echo ""
echo "All repos cloned:"
ls -d "$CODE_DIR"/*/
echo ""
echo "Next step: sbatch $VSC_DATA/scripts/02_install_unified.sh"
