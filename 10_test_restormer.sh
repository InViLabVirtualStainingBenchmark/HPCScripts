#!/bin/bash

#SBATCH --job-name=test_restormer_BCI

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH -A ap_invilab

#SBATCH -p ampere_gpu

#SBATCH --gres=gpu:1

#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/test_restormer_%j.out

#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/test_restormer_%j.err



set -euo pipefail



DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/Restormer"

VENV_DIR="$VSC_DATA/projects/unified/venv"

DATA_BASE="$VSC_SCRATCH/unified/model_data/restormer/$DATASET"

OUT_DIR="$VSC_DATA/projects/unified/outputs/restormer/${DATASET}_1epoch"

WEIGHTS="$OUT_DIR/checkpoints/net_g_latest.pth"

RESULTS_DIR="$OUT_DIR/results"



mkdir -p "$RESULTS_DIR"



module purge

module load calcua/2023a

module load SciPy-bundle/2023.07-gfbf-2023a

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"



python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"



echo "=========================================="

echo "  Restormer Inference — $DATASET"

echo "  Weights : $WEIGHTS"

echo "  Results : $RESULTS_DIR"

echo "=========================================="



# Write Python test script to file first

PYFILE="$OUT_DIR/run_test_restormer.py"

cat > "$PYFILE" << PYEOF

import torch

import torch.nn.functional as F

import os, sys, cv2, numpy as np

from tqdm import tqdm

from skimage import img_as_ubyte

from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from skimage.metrics import structural_similarity as ssim_fn

from natsort import natsorted

from glob import glob

from runpy import run_path



repo_dir = sys.argv[1]

weights  = sys.argv[2]

input_dir = sys.argv[3]

gt_dir   = sys.argv[4]

result_dir = sys.argv[5]



os.makedirs(result_dir, exist_ok=True)



load_arch = run_path(os.path.join(repo_dir, 'basicsr', 'models', 'archs', 'restormer_arch.py'))

Restormer = load_arch['Restormer']



parameters = {

    'inp_channels': 3, 'out_channels': 3, 'dim': 48,

    'num_blocks': [4,6,6,8], 'num_refinement_blocks': 4,

    'heads': [1,2,4,8], 'ffn_expansion_factor': 2.66,

    'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False

}



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Restormer(**parameters)

model.to(device)

checkpoint = torch.load(weights, map_location=device)

model.load_state_dict(checkpoint['params'])

model.eval()

print('Model loaded successfully')



files = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

print(f'Found {len(files)} test images')



psnr_list, ssim_list = [], []

img_multiple_of = 8



with torch.no_grad():

    for file_ in tqdm(files):

        fname = os.path.splitext(os.path.basename(file_))[0]

        img = cv2.cvtColor(cv2.imread(file_), cv2.COLOR_BGR2RGB)

        gt_path = os.path.join(gt_dir, os.path.basename(file_))

        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)



        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        h, w = input_.shape[2], input_.shape[3]

        H = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of

        W = ((w+img_multiple_of)//img_multiple_of)*img_multiple_of

        padh = H-h if h%img_multiple_of!=0 else 0

        padw = W-w if w%img_multiple_of!=0 else 0

        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')



        restored = model(input_)

        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0,2,3,1).cpu().detach().numpy()

        restored = img_as_ubyte(restored[0])



        cv2.imwrite(os.path.join(result_dir, fname+'.png'), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))



        restored_f = restored.astype(np.float32) / 255.

        gt_f = gt.astype(np.float32) / 255.

        psnr = psnr_fn(gt_f, restored_f, data_range=1.0)

        ssim = ssim_fn(gt_f, restored_f, channel_axis=2, data_range=1.0)

        psnr_list.append(psnr)

        ssim_list.append(ssim)



print(f'Mean PSNR: {np.mean(psnr_list):.4f} dB')

print(f'Mean SSIM: {np.mean(ssim_list):.4f}')

with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:

    f.write(f'Mean PSNR: {np.mean(psnr_list):.4f} dB')

    f.write(f'Mean SSIM: {np.mean(ssim_list):.4f}')

PYEOF



cd "$REPO_DIR"

python "$PYFILE" "$REPO_DIR" "$WEIGHTS" "$DATA_BASE/test/input" "$DATA_BASE/test/target" "$RESULTS_DIR"



echo "Done. Results saved to: $RESULTS_DIR"

deactivate

