# HPC Scripts — Virtual Staining Benchmark

## Project Overview

The Virtual Staining Benchmark evaluates five image restoration models adapted for H&E to IHC virtual staining across multiple pathology datasets. This script-based pipeline runs on the VSC Tier-2 cluster using SLURM for job scheduling.

**Models benchmarked:**
- Uformer
- SwinIR (via KAIR framework)
- Restormer
- NAFNet
- HAT

**Datasets used:**
- BCI (Breast Cancer Immunohistochemistry)
- MIST (ER, HER2, Ki67, PR)  (To be done)
- DeepLIIF (To be done )
- ORION-CRC (To be done )

---

## Cluster Infrastructure

| Cluster | GPU | Partition | Best For |
|---|---|---|---|
| Vaughan | NVIDIA A100 (80 GB) | ampere_gpu | Large models — HAT, SwinIR, NAFNet |
| Leibniz | NVIDIA P100 (16 GB) | pascal_gpu | Smaller models — Uformer, Restormer |

---

## Repository Structure

```
HPCScripts/
├── 00_setup_project.sh         # One-time project setup
├── 01_clone_repos.sh           # Clone all 5 model repositories
├── 02_install_unified.sh       # Install shared Python virtual environment
├── 03_prepare_datasets.sh      # Verify dataset structure and paired counts
├── 04_train_uformer.sh         # Train Uformer on BCI
├── 05_train_kair_swinir.sh     # Train SwinIR via KAIR on BCI
├── 06_train_restormer.sh       # Train Restormer on BCI
├── 07_train_nafnet.sh          # Train NAFNet on BCI
├── 08_train_hat.sh             # Train HAT on BCI (reduced batch)
├── 09_evaluate_all.sh          # Compute PSNR and SSIM for all models
├── 10_test_restormer.sh        # Restormer inference on BCI test set
├── 11_test_uformer.sh          # Uformer inference on BCI test set
├── 12_test_nafnet.sh           # NAFNet inference on BCI test set
├── 13_test_hat.sh              # HAT inference on BCI test set (tiled)
└── 14_test_kair_swinir.sh      # SwinIR inference on BCI test set
```

---

## Prerequisites

- VSC account with access to the `ap_invilab` allocation
- SSH access to `login1-vaughan.hpc.uantwerpen.be`
- Access to the InViLab GitHub organisation repositories

---

## First-Time Setup

Run these steps once before submitting any training jobs:

```bash
# 1. Clone this repository
git clone https://github.com/InViLabVirtualStainingBenchmark/HPCScripts.git $VSC_DATA/scripts/

# 2. Set up the project directories
sbatch $VSC_DATA/scripts/00_setup_project.sh

# 3. Clone all model repositories
sbatch $VSC_DATA/scripts/01_clone_repos.sh

# 4. Install the shared virtual environment
sbatch $VSC_DATA/scripts/02_install_unified.sh

# 5. Verify dataset structure
sbatch $VSC_DATA/scripts/03_prepare_datasets.sh
```

---

## Running Training Jobs

Each training script follows the same structure. Submit using sbatch:

```bash
sbatch $VSC_DATA/scripts/07_train_nafnet.sh
```

Monitor the job:

```bash
squeue -u $USER
tail -f $VSC_DATA/projects/unified/logs/train_nafnet_<jobid>.out
```

Check disk quota before every submission:

```bash
myquota
```

---

## Running Evaluation

After all 5 models have completed inference, run the unified evaluation script:

```bash
sbatch $VSC_DATA/scripts/09_evaluate_all.sh BCI
```

Results are saved to:  $VSC_DATA/projects/unified/outputs/eval_results/BCI_metrics.csv

---

## Storage Layout

| Location | Purpose |
|---|---|
| `$VSC_DATA/scripts/` | All SLURM scripts (this repository) |
| `$VSC_DATA/projects/unified/code/` | Model source repositories |
| `$VSC_DATA/projects/unified/venv/` | Shared Python virtual environment |
| `$VSC_DATA/projects/unified/outputs/` | Checkpoints, inference results, evaluation CSV |
| `$VSC_DATA/projects/unified/logs/` | SLURM .out and .err log files |
| `$VSC_SCRATCH/unified/model_data/` | Training datasets |

---

## Module Loading

All scripts load modules in this exact order:

```bash
module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source $VSC_DATA/projects/unified/venv/bin/activate
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Job pending forever | `sinfo \| grep idle` — try a different partition |
| CUDA out of memory | Reduce `batch_size` and `gt_size` in config |
| Dataset not found | Verify path with `ls $VSC_SCRATCH/...` |
| Disk quota exceeded | `myquota` then remove old checkpoints |
| ImportError basicsr | Set `PYTHONPATH=$REPO_DIR` before python command |

For detailed troubleshooting see the system documentation Section 14, 15, & 16.
