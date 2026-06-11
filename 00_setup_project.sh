#!/bin/bash
# =============================================================
# 00_setup_project.sh
# Run ONCE on the login node to create the full project layout.
# Usage: bash 00_setup_project.sh
# =============================================================
# Models: Uformer, KAIR/SwinIR, Restormer, NAFNet, HAT
# Dataset: BCI only
# =============================================================

BASE_DATA="$VSC_DATA/projects/unified"
BASE_SCRATCH="$VSC_SCRATCH/unified"

echo "=========================================="
echo "  Setting up Unified Model Benchmark"
echo "  Models : Uformer, KAIR/SwinIR, Restormer, NAFNet, HAT"
echo "  Dataset: BCI"
echo "  DATA   : $BASE_DATA"
echo "  SCRATCH: $BASE_SCRATCH"
echo "=========================================="

# ── VSC_DATA (persistent storage) ──────────────────────────
mkdir -p "$BASE_DATA"/{code,jobs,logs,venv}

# One output folder per model
for model in uformer kair_swinir restormer nafnet hat; do
    mkdir -p "$BASE_DATA/outputs/$model"/{checkpoints,results}
done

# Timing logs folder
mkdir -p "$BASE_DATA/outputs/timing"

# ── VSC_SCRATCH (fast I/O during jobs) ─────────────────────
# Per-model symlinked data views (created by 03_prepare_datasets.sh)
for model in uformer kair_swinir restormer nafnet hat; do
    mkdir -p "$BASE_SCRATCH/model_data/$model/BCI"/{train,val,test}/{input,target}
done

echo ""
echo "Done. Layout:"
echo ""
echo "  \$VSC_DATA/projects/unified/"
echo "  ├── code/          <- git clones"
echo "  ├── jobs/          <- job scripts"
echo "  ├── logs/          <- SLURM .out/.err"
echo "  ├── venv/          <- shared virtualenv"
echo "  └── outputs/"
echo "      ├── uformer/"
echo "      ├── kair_swinir/"
echo "      ├── restormer/"
echo "      ├── nafnet/"
echo "      ├── hat/"
echo "      └── timing/    <- training time logs per model"
echo ""
echo "  \$VSC_SCRATCH/unified/model_data/{model}/BCI/{train,val,test}/{input,target}"
echo ""
echo "BCI raw data expected at:"
echo "  \$VSC_SCRATCH/BCI/HE/{train,test}   <- input images"
echo "  \$VSC_SCRATCH/BCI/IHC/{train,test}  <- ground truth images"
echo ""
echo "Next step: bash 01_clone_repos.sh"
