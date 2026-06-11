#!/bin/bash
#SBATCH --job-name=train_nafnet_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/nafnet_%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/nafnet_%j.err

# =============================================================
# 07_train_nafnet.sh — BCI dataset, 1 epoch
# Submit: sbatch 07_train_nafnet.sh BCI
# =============================================================
# NAFNET NOTES FROM DOCUMENTATION:
# - Does NOT support --auto_resume flag (removed)
# - Uses python basicsr/train.py (NAFNet's own basicsr)
# - Optimizer: AdamW, lr=1e-3, betas=[0.9, 0.9]
# - Loss: PSNRLoss
# - Scheduler: TrueCosineAnnealingLR
# - dist_params backend: nccl on HPC
# - Val section removed to avoid 977/978 mismatch crash
# =============================================================

set -euo pipefail

DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/NAFNet"
VENV_DIR="$VSC_DATA/projects/unified/venv"
DATA_BASE="$VSC_SCRATCH/unified/model_data/nafnet/$DATASET"
OUT_DIR="$VSC_DATA/projects/unified/outputs/nafnet/${DATASET}_1epoch"
TIMING_LOG="$VSC_DATA/projects/unified/outputs/timing/training_times.csv"

TRAIN_INPUT="$DATA_BASE/train/input"
TRAIN_TARGET="$DATA_BASE/train/target"

BATCH_SIZE=4

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "  NAFNet Training — $DATASET (1 epoch)"
echo "  Data    : $DATA_BASE"
echo "  Output  : $OUT_DIR"
echo "=========================================="

[ -d "$REPO_DIR" ]    || { echo "ERROR: NAFNet repo missing."; deactivate; exit 1; }
[ -d "$TRAIN_INPUT" ] || { echo "ERROR: Data missing. Run 03_prepare_datasets.sh"; deactivate; exit 1; }

python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

TRAIN_COUNT=$(find "$TRAIN_INPUT" -maxdepth 1 \( -type f -o -type l \) | wc -l)
ITERS_1EPOCH=$(( (TRAIN_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Train images : $TRAIN_COUNT | 1 epoch = $ITERS_1EPOCH iters"

mkdir -p "$OUT_DIR"/{checkpoints,logs,visualization}
CONFIG_FILE="$OUT_DIR/train_nafnet_${DATASET}.yml"

cat > "$CONFIG_FILE" << YAML

name: nafnet_${DATASET}

model_type: ImageRestorationModel

scale: 1

num_gpu: 1

manual_seed: 100

datasets:

  train:

    name: TrainSet

    type: PairedImageDataset

    dataroot_gt: ${TRAIN_TARGET}

    dataroot_lq: ${TRAIN_INPUT}

    geometric_augs: true

    use_flip: true

    use_rot: true

    filename_tmpl: '{}'

    io_backend:

      type: disk

    num_worker_per_gpu: 4

    batch_size_per_gpu: ${BATCH_SIZE}

    gt_size: 256

    dataset_enlarge_ratio: 1

    prefetch_mode: ~

  val:

    name: ValSet

    type: PairedImageDataset

    dataroot_gt: ${DATA_BASE}/val/target

    dataroot_lq: ${DATA_BASE}/val/input

    io_backend:

      type: disk

network_g:

  type: NAFNet

  width: 64

  enc_blk_nums: [2, 2, 4, 8]

  middle_blk_num: 12

  dec_blk_nums: [2, 2, 2, 2]

path:

  pretrain_network_g: ~

  strict_load_g: true

  resume_state: ~

  experiments_root: ${OUT_DIR}

  models: ${OUT_DIR}/checkpoints

  training_states: ${OUT_DIR}/checkpoints

  log: ${OUT_DIR}/logs

  visualization: ${OUT_DIR}/visualization

train:

  total_iter: ${ITERS_1EPOCH}

  warmup_iter: -1

  use_grad_clip: true

  scheduler:

    type: TrueCosineAnnealingLR

    T_max: ${ITERS_1EPOCH}

    eta_min: !!float 1e-7

  optim_g:

    type: AdamW

    lr: !!float 1e-3

    weight_decay: !!float 1e-3

    betas: [0.9, 0.9]

  pixel_opt:

    type: PSNRLoss

    loss_weight: 1

    reduction: mean

val:

  val_freq: !!float 1e9

  save_img: false

  metrics:

    psnr:

      type: calculate_psnr

      crop_border: 0

      test_y_channel: false

logger:

  print_freq: 100

  save_checkpoint_freq: ${ITERS_1EPOCH}

  use_tb_logger: true

  wandb:

    project: ~

dist_params:

  backend: nccl

  port: 29500

YAML


nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 10 > "$OUT_DIR/gpu_usage.csv" &
GPU_LOG_PID=$!

START_TIME=$(date +%s)
START_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "Training started: $START_STR"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# NOTE: NAFNet does NOT support --auto_resume
python basicsr/train.py \
    -opt "$CONFIG_FILE" \
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
echo "nafnet,$DATASET,$START_STR,$END_STR,$ELAPSED,$DURATION,$TRAIN_COUNT" >> "$TIMING_LOG"
echo "Timing saved to: $TIMING_LOG"

kill $GPU_LOG_PID || true
deactivate
