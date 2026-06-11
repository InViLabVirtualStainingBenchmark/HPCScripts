#!/bin/bash
# =============================================================
# 03_prepare_datasets.sh
# Run ONCE on the login node (NOT sbatch) after uploading data.
# Usage: bash 03_prepare_datasets.sh
# =============================================================
# NOTE ON UFORMER DATA STRUCTURE:
# Uformer's dataset_denoise.py hardcodes 'groundtruth' as the GT
# subfolder name. All other models use 'target'. This script
# creates the correct subfolder names per model.
# =============================================================

set -euo pipefail

RAW="$VSC_SCRATCH"
OUT="$VSC_SCRATCH/unified/model_data"

echo "=========================================="
echo "  Preparing BCI dataset for all 5 models"
echo "  RAW   : $RAW/BCI"
echo "  OUTPUT: $OUT"
echo "=========================================="

for d in BCI/HE/train BCI/HE/test BCI/IHC/train BCI/IHC/test; do
    if [ ! -d "$RAW/$d" ]; then
        echo "ERROR: Missing: $RAW/$d"; exit 1
    fi
done

link_files() {
    local src="$1" dst="$2" label="$3" count=0
    mkdir -p "$dst"
    for f in "$src"/*; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$dst/"
        count=$((count + 1))
    done
    echo "    $label: $count files linked"
}

# ── Restormer, NAFNet, HAT: use input/ and target/ ──────────
for model in restormer nafnet hat; do
    echo ""
    echo "  Linking BCI for $model..."
    link_files "$RAW/BCI/HE/train"  "$OUT/$model/BCI/train/input"  "train/input"
    link_files "$RAW/BCI/IHC/train" "$OUT/$model/BCI/train/target" "train/target"
    link_files "$RAW/BCI/HE/test"   "$OUT/$model/BCI/test/input"   "test/input"
    link_files "$RAW/BCI/IHC/test"  "$OUT/$model/BCI/test/target"  "test/target"
done

# ── KAIR/SwinIR: uses input/ and target/ ────────────────────
echo ""
echo "  Linking BCI for kair_swinir..."
link_files "$RAW/BCI/HE/train"  "$OUT/kair_swinir/BCI/train/input"  "train/input"
link_files "$RAW/BCI/IHC/train" "$OUT/kair_swinir/BCI/train/target" "train/target"
link_files "$RAW/BCI/HE/test"   "$OUT/kair_swinir/BCI/test/input"   "test/input"
link_files "$RAW/BCI/IHC/test"  "$OUT/kair_swinir/BCI/test/target"  "test/target"

# ── Uformer: MUST use input/ and groundtruth/ ───────────────
# Uformer's dataset_denoise.py hardcodes 'groundtruth' as GT folder
echo ""
echo "  Linking BCI for uformer..."
link_files "$RAW/BCI/HE/train"  "$OUT/uformer/BCI/train/input"       "train/input"
link_files "$RAW/BCI/IHC/train" "$OUT/uformer/BCI/train/groundtruth" "train/groundtruth"
link_files "$RAW/BCI/HE/test"   "$OUT/uformer/BCI/test/input"        "test/input"
link_files "$RAW/BCI/IHC/test"  "$OUT/uformer/BCI/test/groundtruth"  "test/groundtruth"

# ── Count verification ───────────────────────────────────────
echo ""
echo "── Count verification ───────────────────"
for model in restormer nafnet hat kair_swinir; do
    TRAIN_IN=$(find "$OUT/$model/BCI/train/input"  -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)
    TRAIN_GT=$(find "$OUT/$model/BCI/train/target" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)
    echo "  $model — train: $TRAIN_IN input / $TRAIN_GT target"
done
UF_IN=$(find "$OUT/uformer/BCI/train/input"       -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)
UF_GT=$(find "$OUT/uformer/BCI/train/groundtruth" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)
echo "  uformer  — train: $UF_IN input / $UF_GT groundtruth"

echo ""
echo "BCI dataset ready for all 5 models."
