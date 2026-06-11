#!/bin/bash

#SBATCH --job-name=test_hat_BCI

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=08:00:00

#SBATCH -A ap_invilab

#SBATCH -p ampere_gpu

#SBATCH --gres=gpu:1

#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/test_hat_%j.out

#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/test_hat_%j.err



set -euo pipefail



DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/HAT"

VENV_DIR="$VSC_DATA/projects/unified/venv"

DATA_BASE="$VSC_SCRATCH/unified/model_data/hat/$DATASET"

OUT_DIR="$VSC_DATA/projects/unified/outputs/hat/${DATASET}_1epoch"

WEIGHTS="$OUT_DIR/checkpoints/net_g_latest.pth"

RESULTS_DIR="$OUT_DIR/results"



mkdir -p "$RESULTS_DIR"



module purge

module load calcua/2023a

module load SciPy-bundle/2023.07-gfbf-2023a

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"



python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"



CONFIG_FILE="$OUT_DIR/test_hat_${DATASET}.yml"

cat > "$CONFIG_FILE" << YAML

name: hat_${DATASET}_test

model_type: HATModel

scale: 1

num_gpu: 1

manual_seed: 0

tile:

  tile_size: 256

  tile_pad: 32

datasets:

  test:

    name: ${DATASET}_test

    type: PairedImageDataset

    dataroot_gt: ${DATA_BASE}/test/target

    dataroot_lq: ${DATA_BASE}/test/input

    io_backend:

      type: disk

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

  pretrain_network_g: ${WEIGHTS}

  strict_load_g: true

  param_key_g: params

  results_root: ${RESULTS_DIR}

  log: ${OUT_DIR}/logs

  visualization: ${RESULTS_DIR}

val:

  save_img: true

  suffix: ~

  metrics:

    psnr:

      type: calculate_psnr

      crop_border: 0

      test_y_channel: false

    ssim:

      type: calculate_ssim

      crop_border: 0

      test_y_channel: false

dist_params:

  backend: nccl

  port: 29500

YAML



echo "=========================================="

echo "  HAT Inference — $DATASET"

echo "  Weights : $WEIGHTS"

echo "  Results : $RESULTS_DIR"

echo "=========================================="



cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

python hat/test.py -opt "$CONFIG_FILE"



echo "Done. Results saved to: $RESULTS_DIR"

deactivate

