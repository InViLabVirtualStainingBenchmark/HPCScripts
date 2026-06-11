#!/bin/bash
#SBATCH --job-name=train_hat_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/hat_%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/hat_%j.err

# =============================================================
# 08_train_hat.sh — BCI dataset, 1 epoch
# Submit: sbatch 08_train_hat.sh BCI
# =============================================================
# HAT HPC SETTINGS FROM DOCUMENTATION:
# - window_size: 16 (not 8 — HPC setting)
# - embed_dim: 180 (not 96 — HPC setting)
# - depths: [6,6,6,6,6,6] (not [4,4,4,4])
# - num_heads: [6,6,6,6,6,6]
# - compress_ratio: 3 (not 24)
# - squeeze_factor: 30 (not 24)
# - gt_size: 128
# - ema_decay: 0 (avoids gradient issues)
# - upsampler: '' (scale=1, virtual staining — no upsampling)
# - Val section removed to avoid 977/978 mismatch crash
# =============================================================

set -euo pipefail

DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/HAT"
VENV_DIR="$VSC_DATA/projects/unified/venv"
DATA_BASE="$VSC_SCRATCH/unified/model_data/hat/$DATASET"
OUT_DIR="$VSC_DATA/projects/unified/outputs/hat/${DATASET}_1epoch"
TIMING_LOG="$VSC_DATA/projects/unified/outputs/timing/training_times.csv"

TRAIN_INPUT="$DATA_BASE/train/input"
TRAIN_TARGET="$DATA_BASE/train/target"

BATCH_SIZE=2

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "  HAT Training — $DATASET (1 epoch)"
echo "  Data    : $DATA_BASE"
echo "  Output  : $OUT_DIR"
echo "=========================================="

[ -d "$REPO_DIR" ]    || { echo "ERROR: HAT repo missing."; deactivate; exit 1; }
[ -d "$TRAIN_INPUT" ] || { echo "ERROR: Data missing. Run 03_prepare_datasets.sh"; deactivate; exit 1; }

python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

TRAIN_COUNT=$(find "$TRAIN_INPUT" -maxdepth 1 \( -type f -o -type l \) | wc -l)
ITERS_1EPOCH=$(( (TRAIN_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Train images : $TRAIN_COUNT | 1 epoch = $ITERS_1EPOCH iters"

mkdir -p "$OUT_DIR"/{checkpoints,logs,visualization}
CONFIG_FILE="$OUT_DIR/train_hat_${DATASET}.yml"

cat > "$CONFIG_FILE" << YAML
name: hat_${DATASET}
model_type: HATModel
scale: 1
num_gpu: 1
manual_seed: 100

datasets:
  train:
    name: TrainSet
    type: PairedImageDataset
    dataroot_gt: ${TRAIN_TARGET}
    dataroot_lq: ${TRAIN_INPUT}
    filename_tmpl: '{}'
    io_backend:
      type: disk
    gt_size: 128
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: ${BATCH_SIZE}
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

network_g:
  type: HAT
  upscale: 1
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.0
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: ''
  resi_connection: '1conv'

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
  ema_decay: 0
  scheduler:
    type: MultiStepLR
    milestones: [${ITERS_1EPOCH}]
    gamma: 0.5
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
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

python hat/train.py \
    -opt "$CONFIG_FILE" \
    --auto_resume \
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
echo "hat,$DATASET,$START_STR,$END_STR,$ELAPSED,$DURATION,$TRAIN_COUNT" >> "$TIMING_LOG"
echo "Timing saved to: $TIMING_LOG"

kill $GPU_LOG_PID || true
deactivate
