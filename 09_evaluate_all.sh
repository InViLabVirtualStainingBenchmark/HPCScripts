#!/bin/bash
#SBATCH --job-name=eval_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/eval_%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/eval_%j.err

# =============================================================
# 09_evaluate_all.sh — BCI dataset
# Computes PSNR and SSIM for all trained models.
# Also prints training time comparison table.
# Submit: sbatch 09_evaluate_all.sh BCI
# =============================================================

set -euo pipefail

DATASET="${1:-BCI}"

VENV_DIR="$VSC_DATA/projects/unified/venv"
RESULTS_BASE="$VSC_DATA/projects/unified/outputs"
GT_BASE="$VSC_SCRATCH/unified/model_data"
EVAL_DIR="$VSC_DATA/projects/unified/outputs/eval_results"
TIMING_LOG="$VSC_DATA/projects/unified/outputs/timing/training_times.csv"
RESULTS_CSV="$EVAL_DIR/${DATASET}_metrics.csv"

mkdir -p "$EVAL_DIR"
rm -f "$RESULTS_CSV"

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ── Evaluation script ────────────────────────────────────────
EVAL_SCRIPT="$EVAL_DIR/compute_metrics.py"

cat > "$EVAL_SCRIPT" << 'PYEOF'
import argparse, os, csv
import numpy as np
from skimage import io, metrics

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def load_img(path):
    img = io.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[:, :, :3]

def compute(pred_path, gt_path):
    pred = load_img(pred_path)
    gt   = load_img(gt_path)
    if pred.shape != gt.shape:
        from skimage.transform import resize
        pred = resize(pred, gt.shape, anti_aliasing=True)
    psnr = metrics.peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = metrics.structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
    return psnr, ssim

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred",    required=True)
    p.add_argument("--gt",      required=True)
    p.add_argument("--model",   required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out",     required=True)
    args = p.parse_args()

    gt_files = sorted(f for f in os.listdir(args.gt)
                      if os.path.splitext(f)[1].lower() in EXTS)
    psnr_list, ssim_list, missing = [], [], 0
    for fname in gt_files:
        pred_path = os.path.join(args.pred, fname)
        if not os.path.isfile(pred_path):
            missing += 1
            continue
        psnr, ssim = compute(pred_path, os.path.join(args.gt, fname))
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    if missing:
        print(f"WARNING: {missing} images had no prediction.")
    if not psnr_list:
        print("ERROR: No matched pairs found."); return

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"  Evaluated : {len(psnr_list)} images")
    print(f"  Mean PSNR : {mean_psnr:.4f} dB")
    print(f"  Mean SSIM : {mean_ssim:.4f}")

    header = not os.path.isfile(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["model", "dataset", "n_images", "PSNR_dB", "SSIM"])
        w.writerow([args.model, args.dataset, len(psnr_list),
                    f"{mean_psnr:.4f}", f"{mean_ssim:.4f}"])

if __name__ == "__main__":
    main()
PYEOF

GT_DIR="$GT_BASE/restormer/$DATASET/test/target"
[ -d "$GT_DIR" ] || { echo "ERROR: GT not found: $GT_DIR"; deactivate; exit 1; }
echo "GT images: $(find "$GT_DIR" -maxdepth 1 \( -type f -o -type l \) | wc -l)"

# ── Evaluate each model ──────────────────────────────────────
for MODEL in uformer kair_swinir restormer nafnet hat; do
    PRED_DIR="$RESULTS_BASE/$MODEL/${DATASET}_1epoch/results"
    if [ ! -d "$PRED_DIR" ]; then
        echo "SKIP $MODEL — results not found: $PRED_DIR"
        continue
    fi
    echo ""
    echo "Evaluating $MODEL..."
    python "$EVAL_SCRIPT" \
        --pred    "$PRED_DIR" \
        --gt      "$GT_DIR" \
        --model   "$MODEL" \
        --dataset "$DATASET" \
        --out     "$RESULTS_CSV"
done

# ── Print PSNR/SSIM summary ──────────────────────────────────
echo ""
echo "=========================================="
echo "  PSNR / SSIM Results — $DATASET"
echo "=========================================="
if [ -f "$RESULTS_CSV" ]; then
    column -t -s',' "$RESULTS_CSV"
else
    echo "No results generated."
fi

# ── Print training time summary ──────────────────────────────
echo ""
echo "=========================================="
echo "  Training Time Comparison — $DATASET"
echo "=========================================="
if [ -f "$TIMING_LOG" ]; then
    grep ",$DATASET," "$TIMING_LOG" | column -t -s',' || \
    echo "No timing entries found for $DATASET yet."
    echo ""
    echo "Full timing log: $TIMING_LOG"
else
    echo "No timing log found. Run training jobs first."
fi

echo ""
echo "Results CSV : $RESULTS_CSV"
deactivate
