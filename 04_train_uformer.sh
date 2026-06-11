#!/bin/bash
#SBATCH --job-name=train_uformer_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/uformer_%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/uformer_%j.err

# =============================================================
# 04_train_uformer.sh — BCI dataset, 1 epoch
# Submit: sbatch 04_train_uformer.sh BCI
# =============================================================
# UFORMER NOTES FROM DOCUMENTATION:
# - Training script is train/train_denoise.py (NOT train.py)
# - --train_dir points to folder containing input/ and groundtruth/
# - Arg is --train_ps (patch size) NOT --patch_size
# - Needs --embed_dim 32 and --exp_name
# - No --gt_dir argument — groundtruth/ is hardcoded subfolder
# =============================================================

set -euo pipefail

DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/Uformer"
VENV_DIR="$VSC_DATA/projects/unified/venv"
DATA_BASE="$VSC_SCRATCH/unified/model_data/uformer/$DATASET"
OUT_DIR="$VSC_DATA/projects/unified/outputs/uformer/${DATASET}_1epoch"
TIMING_LOG="$VSC_DATA/projects/unified/outputs/timing/training_times.csv"

# Uformer expects --train_dir to contain input/ and groundtruth/ subdirs
TRAIN_DIR="$DATA_BASE/train"
VAL_DIR="$DATA_BASE/test"

ARCH="Uformer_B"
BATCH_SIZE=1
PATCH_SIZE=128
EMBED_DIM=32
NEPOCH=1
LR=0.0002
GPU="0"
TRAIN_WORKERS=4
ENV_NAME="BCI_HE2IHC"

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "  Uformer Training — $DATASET (1 epoch)"
echo "  Script  : train/train_denoise.py"
echo "  Arch    : $ARCH"
echo "  Data    : $DATA_BASE"
echo "  Output  : $OUT_DIR"
echo "=========================================="

[ -f "$REPO_DIR/train/train_denoise.py" ] || { echo "ERROR: train/train_denoise.py not found in Uformer repo."; deactivate; exit 1; }
[ -d "$TRAIN_DIR/input" ]       || { echo "ERROR: Data missing. Run 03_prepare_datasets.sh"; deactivate; exit 1; }
[ -d "$TRAIN_DIR/groundtruth" ] || { echo "ERROR: groundtruth/ folder missing. Run 03_prepare_datasets.sh"; deactivate; exit 1; }

python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
TRAIN_COUNT=$(find "$TRAIN_DIR/input" -maxdepth 1 \( -type f -o -type l \) | wc -l)
echo "Train images : $TRAIN_COUNT"
echo "Val images   : $(find "$VAL_DIR/input" -maxdepth 1 \( -type f -o -type l \) | wc -l)"

mkdir -p "$OUT_DIR"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 10 > "$OUT_DIR/gpu_usage.csv" &
GPU_LOG_PID=$!

START_TIME=$(date +%s)
START_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "Training started: $START_STR"

cd "$REPO_DIR/train"

python train_denoise.py \
    --arch          "$ARCH" \
    --batch_size    "$BATCH_SIZE" \
    --gpu           "$GPU" \
    --train_dir     "$TRAIN_DIR" \
    --val_dir       "$VAL_DIR" \
    --save_dir      "$OUT_DIR" \
    --nepoch        "$NEPOCH" \
    --lr_initial    "$LR" \
    --train_ps      "$PATCH_SIZE" \
    --embed_dim     "$EMBED_DIM" \
    --env           "$ENV_NAME" \
    --train_workers "$TRAIN_WORKERS" \
    --eval_workers  "$TRAIN_WORKERS" \
    2>&1 | tee "$OUT_DIR/train_log.txt"

END_TIME=$(date +%s)
END_STR=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))
DURATION="${HOURS}h ${MINUTES}m ${SECONDS}s"

echo ""
echo "Training finished : $END_STR"
echo "Total duration    : $DURATION ($ELAPSED seconds)"

mkdir -p "$(dirname "$TIMING_LOG")"
if [ ! -f "$TIMING_LOG" ]; then
    echo "model,dataset,start_time,end_time,duration_seconds,duration_human,train_images" > "$TIMING_LOG"
fi
echo "uformer,$DATASET,$START_STR,$END_STR,$ELAPSED,$DURATION,$TRAIN_COUNT" >> "$TIMING_LOG"
echo "Timing saved to: $TIMING_LOG"

kill $GPU_LOG_PID || true
deactivate
