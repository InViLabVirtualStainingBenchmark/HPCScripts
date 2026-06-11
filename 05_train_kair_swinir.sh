#!/bin/bash
#SBATCH --job-name=train_kair_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/kair_swinir_%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/kair_swinir_%j.err

# =============================================================
# 05_train_kair_swinir.sh — BCI dataset, 1 epoch
# Submit: sbatch 05_train_kair_swinir.sh BCI
# =============================================================
# KAIR NOTES FROM DOCUMENTATION:
# - dataset_type must be "plain" for HE2IHC image-to-image task
# - "n_channels": 3 must be at ROOT level of JSON (KAIR broadcasts it)
# - dataroot_L = input (HE), dataroot_H = target (IHC)
# - No --auto_resume flag
# =============================================================

set -euo pipefail

DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/KAIR"
VENV_DIR="$VSC_DATA/projects/unified/venv"
DATA_BASE="$VSC_SCRATCH/unified/model_data/kair_swinir/$DATASET"
OUT_DIR="$VSC_DATA/projects/unified/outputs/kair_swinir/${DATASET}_1epoch"
TIMING_LOG="$VSC_DATA/projects/unified/outputs/timing/training_times.csv"

TRAIN_INPUT="$DATA_BASE/train/input"
TRAIN_TARGET="$DATA_BASE/train/target"
VAL_INPUT="$DATA_BASE/test/input"
VAL_TARGET="$DATA_BASE/test/target"

BATCH_SIZE=8

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "  KAIR/SwinIR Training — $DATASET (1 epoch)"
echo "  Data    : $DATA_BASE"
echo "  Output  : $OUT_DIR"
echo "=========================================="

[ -d "$REPO_DIR" ]    || { echo "ERROR: KAIR repo missing."; deactivate; exit 1; }
[ -d "$TRAIN_INPUT" ] || { echo "ERROR: Data missing. Run 03_prepare_datasets.sh"; deactivate; exit 1; }

python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
TRAIN_COUNT=$(find "$TRAIN_INPUT" -maxdepth 1 \( -type f -o -type l \) | wc -l)
ITERS_1EPOCH=$(( (TRAIN_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Train images : $TRAIN_COUNT | 1 epoch = $ITERS_1EPOCH iters"

mkdir -p "$OUT_DIR"/{checkpoints,results}
CONFIG_FILE="$OUT_DIR/train_swinir_BCI.json"
ORIG_CONFIG="$VSC_DATA/projects/unified/code/KAIR/options/swinir/train_swinir_HE2IHC.json"

python3 -c "
import re, json
with open('$ORIG_CONFIG') as f:
    content = re.sub(r'//.*', '', f.read())
opt = json.loads(content)
opt['path']['root'] = '$OUT_DIR'
opt['datasets']['train']['dataroot_H'] = '$TRAIN_TARGET'
opt['datasets']['train']['dataroot_L'] = '$TRAIN_INPUT'
opt['datasets']['train']['dataloader_batch_size'] = $BATCH_SIZE
opt['datasets']['test']['dataroot_H'] = '$VAL_TARGET'
opt['datasets']['test']['dataroot_L'] = '$VAL_INPUT'
opt['train']['G_scheduler_milestones'] = [$ITERS_1EPOCH]
opt['train']['checkpoint_test'] = $ITERS_1EPOCH
opt['train']['checkpoint_save'] = $ITERS_1EPOCH
opt['train']['G_optimizer_iter'] = $ITERS_1EPOCH
opt['train']['train_iters'] = $ITERS_1EPOCH
opt['val'] = {'val_freq': $ITERS_1EPOCH, 'save_img': False}
import os; os.makedirs('$OUT_DIR', exist_ok=True)
with open('$CONFIG_FILE', 'w') as f:
    json.dump(opt, f, indent=2)
print('Config written')
"
echo "Config written to: $CONFIG_FILE"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 10 > "$OUT_DIR/gpu_usage.csv" &
GPU_LOG_PID=$!

START_TIME=$(date +%s)
START_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "Training started: $START_STR"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

python main_train_psnr.py --opt "$CONFIG_FILE" \
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
echo "kair_swinir,$DATASET,$START_STR,$END_STR,$ELAPSED,$DURATION,$TRAIN_COUNT" >> "$TIMING_LOG"
echo "Timing saved to: $TIMING_LOG"

kill $GPU_LOG_PID || true
deactivate
