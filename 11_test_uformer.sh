#!/bin/bash

#SBATCH --job-name=test_uformer_BCI

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH -A ap_invilab

#SBATCH -p ampere_gpu

#SBATCH --gres=gpu:1

#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/test_uformer_%j.out

#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/test_uformer_%j.err



set -euo pipefail



DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/Uformer"

VENV_DIR="$VSC_DATA/projects/unified/venv"

DATA_BASE="$VSC_SCRATCH/unified/model_data/uformer/$DATASET"

OUT_DIR="$VSC_DATA/projects/unified/outputs/uformer/${DATASET}_1epoch"

WEIGHTS="$OUT_DIR/denoising/SIDD/Uformer_BBCI_HE2IHC/models/model_best.pth"

RESULTS_DIR="$OUT_DIR/results"



mkdir -p "$RESULTS_DIR"



module purge

module load calcua/2023a

module load SciPy-bundle/2023.07-gfbf-2023a

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"



echo "=========================================="

echo "  Uformer Inference — $DATASET"

echo "  Weights : $WEIGHTS"

echo "  Results : $RESULTS_DIR"

echo "=========================================="



python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"



cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

python test/test_gopro_hide.py --input_dir "$DATA_BASE/test" --result_dir "$RESULTS_DIR" --weights "$WEIGHTS" --arch Uformer_B --embed_dim 32 --save_images --gpus 0



echo "Done. Results saved to: $RESULTS_DIR"

deactivate

