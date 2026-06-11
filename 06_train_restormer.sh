#!/bin/bash
#SBATCH --job-name=train_restormer_BCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/restormer_%j.out
#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/restormer_%j.err

# =============================================================
# 06_train_restormer.sh — BCI dataset, 1 epoch
# Submit: sbatch 06_train_restormer.sh BCI
# =============================================================
# NOTE: Val section removed from yml to avoid 977/978 file
# count mismatch crash (BCI IHC/test has 1 extra file vs HE/test)
# =============================================================

set -euo pipefail

DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/Restormer"
VENV_DIR="$VSC_DATA/projects/unified/venv"
DATA_BASE="$VSC_SCRATCH/unified/model_data/restormer/$DATASET"
OUT_DIR="$VSC_DATA/projects/unified/outputs/restormer/${DATASET}_1epoch"
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
echo "  Restormer Training — $DATASET (1 epoch)"
echo "  Data    : $DATA_BASE"
echo "  Output  : $OUT_DIR"
echo "=========================================="

[ -d "$REPO_DIR" ]    || { echo "ERROR: Restormer repo missing."; deactivate; exit 1; }
[ -d "$TRAIN_INPUT" ] || { echo "ERROR: Data missing. Run 03_prepare_datasets.sh"; deactivate; exit 1; }

python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import basicsr; print('basicsr:', basicsr.__version__)"

TRAIN_COUNT=$(find "$TRAIN_INPUT" -maxdepth 1 \( -type f -o -type l \) | wc -l)
ITERS_1EPOCH=$(( (TRAIN_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Train images : $TRAIN_COUNT | 1 epoch = $ITERS_1EPOCH iters"

mkdir -p "$OUT_DIR"/{checkpoints,logs,visualization}
CONFIG_FILE="$OUT_DIR/train_${DATASET}.yml"

cat > "$CONFIG_FILE" << YAML
name: restormer_${DATASET}
model_type: ImageCleanModel
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
    use_hflip: true
    use_rot: true
    filename_tmpl: '{}'
    io_backend:
      type: disk
    num_worker_per_gpu: 4
    batch_size_per_gpu: ${BATCH_SIZE}
    mini_batch_sizes: [${BATCH_SIZE}]
    iters: [${ITERS_1EPOCH}]
    gt_size: 128
    lq_size: 128
    gt_sizes: [128]
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

network_g:
  type: Restormer
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4,6,6,8]
  num_refinement_blocks: 4
  heads: [1,2,4,8]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: BiasFree
  dual_pixel_task: False

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
  mixing_augs:
    mixup: false
  scheduler:
    type: CosineAnnealingRestartCyclicLR
    periods: [${ITERS_1EPOCH}]
    restart_weights: [1]
    eta_mins: [0.000001]
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: !!float 1e-4
    betas: [0.9, 0.999]
  pixel_opt:
    type: L1Loss
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

python -m basicsr.train \
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
echo "restormer,$DATASET,$START_STR,$END_STR,$ELAPSED,$DURATION,$TRAIN_COUNT" >> "$TIMING_LOG"
echo "Timing saved to: $TIMING_LOG"

kill $GPU_LOG_PID || true
deactivate
