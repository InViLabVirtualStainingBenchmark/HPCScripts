#!/bin/bash

#SBATCH --job-name=test_nafnet_BCI

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH -A ap_invilab

#SBATCH -p ampere_gpu

#SBATCH --gres=gpu:1

#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/test_nafnet_%j.out

#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/test_nafnet_%j.err



set -euo pipefail



DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/NAFNet"

VENV_DIR="$VSC_DATA/projects/unified/venv"

DATA_BASE="$VSC_SCRATCH/unified/model_data/nafnet/$DATASET"

OUT_DIR="$VSC_DATA/projects/unified/outputs/nafnet/${DATASET}_1epoch"

WEIGHTS="$REPO_DIR/experiments/nafnet_${DATASET}/models/net_g_latest.pth"

RESULTS_DIR="$OUT_DIR/results"



mkdir -p "$RESULTS_DIR"



module purge

module load calcua/2023a

module load SciPy-bundle/2023.07-gfbf-2023a

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"



python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"



CONFIG_FILE="$OUT_DIR/test_nafnet_${DATASET}.yml"

cat > "$CONFIG_FILE" << YAML

name: nafnet_${DATASET}_test

model_type: ImageRestorationModel

scale: 1

num_gpu: 1

manual_seed: 10

datasets:

  test:

    name: ${DATASET}_test

    type: PairedImageDataset

    dataroot_gt: ${DATA_BASE}/test/target

    dataroot_lq: ${DATA_BASE}/test/input

    io_backend:

      type: disk

network_g:

  type: NAFNet

  width: 64

  enc_blk_nums: [2, 2, 4, 8]

  middle_blk_num: 12

  dec_blk_nums: [2, 2, 2, 2]

path:

  pretrain_network_g: ${WEIGHTS}

  strict_load_g: true

  resume_state: ~

  root: ${OUT_DIR}

  results_root: ${RESULTS_DIR}

  log: ${RESULTS_DIR}

  visualization: ${RESULTS_DIR}/visualization

val:

  save_img: true

  grids: false

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

echo "  NAFNet Inference — $DATASET"

echo "  Weights : $WEIGHTS"

echo "  Results : $RESULTS_DIR"

echo "=========================================="



cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

python basicsr/test.py -opt "$CONFIG_FILE"



echo "Done. Results saved to: $RESULTS_DIR"

deactivate

