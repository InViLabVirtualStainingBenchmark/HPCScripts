#!/bin/bash

#SBATCH --job-name=test_kair_BCI

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH -A ap_invilab

#SBATCH -p ampere_gpu

#SBATCH --gres=gpu:1

#SBATCH -o /data/antwerpen/212/vsc21215/projects/unified/logs/test_kair_%j.out

#SBATCH -e /data/antwerpen/212/vsc21215/projects/unified/logs/test_kair_%j.err



set -euo pipefail



DATASET="${1:-BCI}"

REPO_DIR="$VSC_DATA/projects/unified/code/KAIR"

VENV_DIR="$VSC_DATA/projects/unified/venv"

DATA_BASE="$VSC_SCRATCH/unified/model_data/kair_swinir/$DATASET"

OUT_DIR="$VSC_DATA/projects/unified/outputs/kair_swinir/${DATASET}_1epoch"

WEIGHTS="$OUT_DIR/swinir_HE2IHC/models/487_G.pth"

RESULTS_DIR="$OUT_DIR/results"



mkdir -p "$RESULTS_DIR"



module purge

module load calcua/2023a

module load SciPy-bundle/2023.07-gfbf-2023a

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"



python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"



echo "=========================================="

echo "  KAIR/SwinIR Inference — $DATASET"

echo "  Weights : $WEIGHTS"

echo "  Results : $RESULTS_DIR"

echo "=========================================="



PYFILE="$OUT_DIR/run_test_kair.py"

cat > "$PYFILE" << PYEOF

import torch, numpy as np, cv2, os, sys

from tqdm import tqdm

from natsort import natsorted

from glob import glob

from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from skimage.metrics import structural_similarity as ssim_fn



sys.path.insert(0, sys.argv[1])

from models.network_swinir import SwinIR as net



repo_dir  = sys.argv[1]

weights   = sys.argv[2]

input_dir = sys.argv[3]

gt_dir    = sys.argv[4]

result_dir = sys.argv[5]



os.makedirs(result_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = net(upscale=1, in_chans=3, img_size=128, window_size=8,

            img_range=1., depths=[6,6,6,6,6,6], embed_dim=128,

            num_heads=[4,4,4,4,4,4], mlp_ratio=2,

            upsampler='', resi_connection='1conv')

checkpoint = torch.load(weights, map_location=device)

model.load_state_dict(checkpoint, strict=True)

model.eval()

model.to(device)

print('Model loaded!')



files = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

print(f'Found {len(files)} test images')



psnr_list, ssim_list = [], []

window_size = 8



with torch.no_grad():

    for file_ in tqdm(files):

        fname = os.path.splitext(os.path.basename(file_))[0]

        img_lq = cv2.imread(file_, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        img_gt = cv2.imread(os.path.join(gt_dir, os.path.basename(file_)), cv2.IMREAD_COLOR).astype(np.float32) / 255.



        img_lq_t = torch.from_numpy(np.transpose(img_lq[:,:,[2,1,0]], (2,0,1))).float().unsqueeze(0).to(device)

        _, _, h, w = img_lq_t.size()

        h_pad = (h // window_size + 1) * window_size - h

        w_pad = (w // window_size + 1) * window_size - w

        img_lq_t = torch.cat([img_lq_t, torch.flip(img_lq_t, [2])], 2)[:, :, :h+h_pad, :]

        img_lq_t = torch.cat([img_lq_t, torch.flip(img_lq_t, [3])], 3)[:, :, :, :w+w_pad]



        output = model(img_lq_t)

        output = output[..., :h, :w]

        output = output.data.squeeze().float().cpu().clamp_(0,1).numpy()

        output = np.transpose(output[[2,1,0],:,:], (1,2,0))

        output_uint8 = (output * 255.0).round().astype(np.uint8)

        cv2.imwrite(os.path.join(result_dir, fname+'.png'), output_uint8)



        psnr = psnr_fn(img_gt, output, data_range=1.0)

        ssim = ssim_fn(img_gt, output, channel_axis=2, data_range=1.0)

        psnr_list.append(psnr)

        ssim_list.append(ssim)



print(f'Mean PSNR: {np.mean(psnr_list):.4f} dB')

print(f'Mean SSIM: {np.mean(ssim_list):.4f}')

with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:

    f.write(f'Mean PSNR: {np.mean(psnr_list):.4f} dB')

    f.write(f'Mean SSIM: {np.mean(ssim_list):.4f}')

PYEOF



python "$PYFILE" "$REPO_DIR" "$WEIGHTS" "$DATA_BASE/test/input" "$DATA_BASE/test/target" "$RESULTS_DIR"



echo "Done. Results saved to: $RESULTS_DIR"

deactivate

