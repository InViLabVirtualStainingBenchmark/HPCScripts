"""
Virtual Staining Benchmark — Image Comparison Gallery
Adapted for vsc21215 folder structure on VSC Tier-2

Reads predicted images from:
  $VSC_DATA/projects/unified/outputs/<model>/BCI_1epoch/results/

Ground truth from:
  $VSC_SCRATCH/unified/model_data/restormer/BCI/test/target/

Usage:
    python3 generate_gallery_wesley.py
    python3 generate_gallery_wesley.py --max_images 20   # quick preview
    python3 generate_gallery_wesley.py --out /path/to/gallery.html

View on cluster:
    python3 -m http.server 8080 --directory <gallery_dir>

View on local machine:
    ssh -L 8080:localhost:8080 vsc21215@login1-vaughan.hpc.uantwerpen.be
    Then open: http://localhost:8080/gallery.html
"""

import os
import argparse
import shutil
from pathlib import Path

try:
    from skimage import io
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    import numpy as np
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: scikit-image not available. Metrics will not be computed.")

parser = argparse.ArgumentParser()
parser.add_argument('--vsc_data',    default=os.environ.get('VSC_DATA', '/data/antwerpen/212/vsc21215'))
parser.add_argument('--vsc_scratch', default=os.environ.get('VSC_SCRATCH', '/scratch/antwerpen/212/vsc21215'))
parser.add_argument('--max_images',  default=None, type=int)
parser.add_argument('--out',         default=None)
args = parser.parse_args()

VSC_DATA    = args.vsc_data
VSC_SCRATCH = args.vsc_scratch

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_BASE = f"{VSC_DATA}/projects/unified/outputs"
GT_DIR       = f"{VSC_SCRATCH}/unified/model_data/restormer/BCI/test/target"
INPUT_DIR    = f"{VSC_SCRATCH}/unified/model_data/restormer/BCI/test/input"

OUT_HTML     = args.out or f"{VSC_DATA}/projects/unified/outputs/gallery/gallery.html"
GALLERY_DIR  = os.path.dirname(OUT_HTML)
ASSETS_DIR   = os.path.join(GALLERY_DIR, 'assets')

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = [
    ('nafnet',      'NAFNet',      f"{RESULTS_BASE}/nafnet/BCI_1epoch/results"),
    ('uformer',     'Uformer',     f"{RESULTS_BASE}/uformer/BCI_1epoch/results"),
    ('kair_swinir', 'KAIRSwinIR',  f"{RESULTS_BASE}/kair_swinir/BCI_1epoch/results"),
    ('restormer',   'Restormer',   f"{RESULTS_BASE}/restormer/BCI_1epoch/results"),
    ('hat',         'HAT',         f"{RESULTS_BASE}/hat/BCI_1epoch/results"),
]

MODEL_COLORS = {
    'nafnet':      '#4C78A8',
    'uformer':     '#59A14F',
    'kair_swinir': '#E0AC3B',
    'restormer':   '#9C89C9',
    'hat':         '#E45756',
}

# ── Setup output folders ───────────────────────────────────────────────────────
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, 'input'), exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, 'gt'), exist_ok=True)
for key, _, _ in MODELS:
    os.makedirs(os.path.join(ASSETS_DIR, key), exist_ok=True)

# ── Get image list from GT folder ─────────────────────────────────────────────
print(f"Reading GT images from: {GT_DIR}")
all_images = sorted([
    f for f in os.listdir(GT_DIR)
    if f.lower().endswith(('.png', '.jpg'))
])

if args.max_images:
    all_images = all_images[:args.max_images]

print(f"Found {len(all_images)} images")

# ── Compute metrics and copy assets ───────────────────────────────────────────
print("Computing metrics and copying assets...")

# Store per-image metrics: metrics[model_key][filename] = {psnr, ssim}
metrics = {key: {} for key, _, _ in MODELS}
summaries = {}

for fname in all_images:
    gt_path    = os.path.join(GT_DIR, fname)
    input_path = os.path.join(INPUT_DIR, fname)

    # Copy GT
    dst_gt = os.path.join(ASSETS_DIR, 'gt', fname)
    if not os.path.exists(dst_gt) and os.path.exists(gt_path):
        shutil.copy2(gt_path, dst_gt)

    # Copy input (HE)
    dst_in = os.path.join(ASSETS_DIR, 'input', fname)
    if not os.path.exists(dst_in) and os.path.exists(input_path):
        shutil.copy2(input_path, dst_in)

    # For each model
    for key, name, results_dir in MODELS:
        # Handle both .png and .PNG
        pred_path = os.path.join(results_dir, fname)
        if not os.path.exists(pred_path):
            pred_path = os.path.join(results_dir, fname.replace('.png', '.PNG'))
        if not os.path.exists(pred_path):
            pred_path = os.path.join(results_dir, fname.upper())

        dst_pred = os.path.join(ASSETS_DIR, key, fname)
        if os.path.exists(pred_path):
            if not os.path.exists(dst_pred):
                shutil.copy2(pred_path, dst_pred)

            # Compute metrics
            if METRICS_AVAILABLE and os.path.exists(gt_path):
                try:
                    pred_img = io.imread(pred_path).astype(np.float32) / 255.0
                    gt_img   = io.imread(gt_path).astype(np.float32)   / 255.0
                    if pred_img.ndim == 2:
                        pred_img = np.stack([pred_img]*3, axis=-1)
                    if gt_img.ndim == 2:
                        gt_img = np.stack([gt_img]*3, axis=-1)
                    pred_img = pred_img[:,:,:3]
                    gt_img   = gt_img[:,:,:3]
                    p = psnr_fn(gt_img, pred_img, data_range=1.0)
                    s = ssim_fn(gt_img, pred_img, channel_axis=2, data_range=1.0)
                    metrics[key][fname] = {'psnr': p, 'ssim': s}
                except Exception as e:
                    metrics[key][fname] = {'psnr': None, 'ssim': None}
        else:
            metrics[key][fname] = {'psnr': None, 'ssim': None}

# ── Compute summaries ─────────────────────────────────────────────────────────
for key, name, _ in MODELS:
    psnr_vals = [v['psnr'] for v in metrics[key].values() if v.get('psnr') is not None]
    ssim_vals = [v['ssim'] for v in metrics[key].values() if v.get('ssim') is not None]
    summaries[key] = {
        'psnr': round(sum(psnr_vals)/len(psnr_vals), 4) if psnr_vals else 0,
        'ssim': round(sum(ssim_vals)/len(ssim_vals), 4) if ssim_vals else 0,
        'n':    len(psnr_vals)
    }

print("Summaries:")
for key, name, _ in MODELS:
    s = summaries[key]
    print(f"  {name}: PSNR={s['psnr']:.4f} dB  SSIM={s['ssim']:.4f}  ({s['n']} images)")

# ── Build HTML ────────────────────────────────────────────────────────────────
print("Generating HTML...")

max_psnr = max(s['psnr'] for s in summaries.values()) or 1
max_ssim = max(s['ssim'] for s in summaries.values()) or 1

def bar_chart(title, values, unit, max_val, fmt='.2f'):
    bars = ''
    for key, name, _ in MODELS:
        val = values.get(key, 0)
        pct = (val / max_val * 100) if max_val else 0
        color = MODEL_COLORS.get(key, '#555')
        bars += f'''
      <div class="bar-row">
        <div class="bar-label">{name}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{pct:.1f}%;background:{color}">
            <span class="bar-value">{val:{fmt}} {unit}</span>
          </div>
        </div>
      </div>'''
    return f'<div class="chart"><div class="chart-title">{title}</div>{bars}</div>'

psnr_chart = bar_chart('Avg PSNR ↑', {k: summaries[k]['psnr'] for k,_,_ in MODELS}, 'dB', max_psnr)
ssim_chart = bar_chart('Avg SSIM ↑', {k: summaries[k]['ssim'] for k,_,_ in MODELS}, '',   max_ssim, fmt='.4f')

# Summary table
summary_rows = ''
for key, name, _ in MODELS:
    s = summaries[key]
    best_psnr = s['psnr'] == max(summaries[k]['psnr'] for k,_,_ in MODELS)
    best_ssim = s['ssim'] == max(summaries[k]['ssim'] for k,_,_ in MODELS)
    p_style = 'color:#22863a;font-weight:700' if best_psnr else ''
    s_style = 'color:#22863a;font-weight:700' if best_ssim else ''
    summary_rows += f'''
    <tr>
      <td><span class="model-dot" style="background:{MODEL_COLORS[key]}"></span>{name}</td>
      <td style="{p_style}">{s['psnr']:.4f} dB</td>
      <td style="{s_style}">{s['ssim']:.4f}</td>
      <td>{s['n']}</td>
    </tr>'''

# Image rows
image_rows = ''
for fname in all_images:
    row_psnr = [metrics[key].get(fname, {}).get('psnr') for key,_,_ in MODELS]
    valid_psnr = [v for v in row_psnr if v is not None]
    max_p = max(valid_psnr) if valid_psnr else None
    min_p = min(valid_psnr) if valid_psnr else None

    cells = ''
    # HE input
    cells += f'''
      <div class="img-cell">
        <img src="assets/input/{fname}" loading="lazy" title="H&amp;E Input">
        <div class="img-label">H&amp;E Input</div>
      </div>'''

    # Each model
    for i, (key, name, _) in enumerate(MODELS):
        m = metrics[key].get(fname, {})
        p = m.get('psnr')
        s = m.get('ssim')
        if p is not None:
            color = '#22863a' if p == max_p else ('#cb2431' if p == min_p else '#444')
            metric_str = f'<span style="color:{color};font-weight:600">PSNR: {p:.2f} dB</span><br>SSIM: {s:.4f}'
        else:
            metric_str = '<span style="color:#aaa">—</span>'
        cells += f'''
      <div class="img-cell">
        <img src="assets/{key}/{fname}" loading="lazy" title="{name}">
        <div class="img-label" style="color:{MODEL_COLORS[key]}">{name}</div>
        <div class="metric">{metric_str}</div>
      </div>'''

    # Ground truth
    cells += f'''
      <div class="img-cell">
        <img src="assets/gt/{fname}" loading="lazy" title="IHC Ground Truth">
        <div class="img-label" style="color:#1F3864;font-weight:700">Ground Truth</div>
      </div>'''

    image_rows += f'''
  <div class="image-row" data-id="{fname}">
    <div class="row-id">{fname.replace(".png","").replace(".PNG","")}</div>
    <div class="row-panels">{cells}</div>
  </div>'''

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Virtual Staining Benchmark — BCI Gallery</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; background: #f5f6fa; color: #222; padding: 28px; }}

h1 {{ font-size: 1.5rem; color: #1F3864; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: 0.88rem; margin-bottom: 24px; }}

.section {{ background: #fff; border-radius: 10px; padding: 20px 24px;
            margin-bottom: 24px; box-shadow: 0 1px 5px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.07em;
               color: #888; margin-bottom: 16px; }}

.charts-row {{ display: flex; gap: 32px; flex-wrap: wrap; }}
.chart {{ flex: 1; min-width: 240px; }}
.chart-title {{ font-size: 0.82rem; font-weight: 700; color: #444;
                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }}
.bar-row {{ display: flex; align-items: center; margin-bottom: 10px; gap: 12px; }}
.bar-label {{ width: 100px; font-size: 0.82rem; font-weight: 600; color: #333;
              text-align: right; flex-shrink: 0; }}
.bar-track {{ flex: 1; background: #edf2f7; border-radius: 5px; height: 28px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 5px; display: flex; align-items: center; min-width: 50px; }}
.bar-value {{ font-size: 0.74rem; font-weight: 700; color: #fff; padding: 0 10px; white-space: nowrap; }}

table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ padding: 9px 14px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
th {{ background: #f8f9fb; font-weight: 600; color: #555; font-size: 0.8rem;
      text-transform: uppercase; letter-spacing: 0.04em; }}
.model-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
              margin-right: 8px; vertical-align: middle; }}

.filter-bar {{ margin-bottom: 14px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
.filter-bar input {{ padding: 8px 14px; border: 1px solid #ddd; border-radius: 6px;
                     font-size: 0.88rem; width: 260px; }}
.filter-bar label {{ font-size: 0.82rem; color: #666; }}
.filter-bar select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px;
                      font-size: 0.88rem; }}

.image-row {{ background: #fff; border-radius: 8px; margin-bottom: 14px;
              padding: 12px 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.row-id {{ font-size: 0.72rem; color: #aaa; margin-bottom: 8px; font-family: monospace; }}
.row-panels {{ display: flex; gap: 10px; flex-wrap: nowrap; overflow-x: auto; }}

.img-cell {{ text-align: center; flex: 0 0 auto; width: 150px; }}
.img-cell img {{ width: 150px; height: 150px; object-fit: cover;
                 border-radius: 4px; border: 1px solid #e0e0e0;
                 cursor: zoom-in; transition: transform 0.2s;
                 transform-origin: top center; display: block; margin: 0 auto; }}
.img-cell img:hover {{ transform: scale(2.8); z-index: 999;
                        position: relative; box-shadow: 0 8px 28px rgba(0,0,0,0.25); }}
.img-label {{ font-size: 0.72rem; font-weight: 600; color: #555; margin-top: 5px; }}
.metric {{ font-size: 0.68rem; color: #666; margin-top: 3px; line-height: 1.5; }}

.count-badge {{ background: #1F3864; color: #fff; font-size: 0.75rem;
                padding: 3px 10px; border-radius: 12px; margin-left: 8px; }}
</style>
</head>
<body>

<h1>🔬 Virtual Staining Benchmark — BCI Dataset</h1>
<p class="subtitle">H&amp;E → IHC · {len(all_images)} test images · InViLab, University of Antwerp · vsc21215</p>

<div class="section">
  <h2>Model Performance Summary</h2>
  <div class="charts-row">
    {psnr_chart}
    {ssim_chart}
  </div>
</div>

<div class="section">
  <h2>Results Table</h2>
  <table>
    <tr><th>Model</th><th>PSNR (dB) ↑</th><th>SSIM ↑</th><th>Images Evaluated</th></tr>
    {summary_rows}
  </table>
</div>

<div class="section">
  <h2>Per-Image Visual Comparison
    <span class="count-badge">{len(all_images)} images</span>
  </h2>
  <div class="filter-bar">
    <input type="text" id="filterInput" placeholder="Filter by image ID..." oninput="filterRows()">
    <label>Sort by:
      <select onchange="sortRows(this.value)">
        <option value="default">Default</option>
        <option value="psnr_best">Best PSNR (NAFNet)</option>
        <option value="psnr_worst">Worst PSNR (NAFNet)</option>
      </select>
    </label>
  </div>
  <div id="imageGrid">
    {image_rows}
  </div>
</div>

<script>
function filterRows() {{
  const q = document.getElementById('filterInput').value.toLowerCase();
  document.querySelectorAll('.image-row').forEach(row => {{
    row.style.display = row.dataset.id.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function sortRows(mode) {{
  const grid = document.getElementById('imageGrid');
  const rows = Array.from(grid.querySelectorAll('.image-row'));
  if (mode === 'default') {{
    rows.sort((a,b) => a.dataset.id.localeCompare(b.dataset.id));
  }} else {{
    rows.sort((a,b) => {{
      const pa = parseFloat(a.querySelector('.metric span')?.textContent) || 0;
      const pb = parseFloat(b.querySelector('.metric span')?.textContent) || 0;
      return mode === 'psnr_best' ? pb - pa : pa - pb;
    }});
  }}
  rows.forEach(r => grid.appendChild(r));
}}
</script>

</body>
</html>'''

with open(OUT_HTML, 'w') as f:
    f.write(html)

print(f"\nDone!")
print(f"Gallery saved to: {OUT_HTML}")
print(f"\nTo view:")
print(f"  On cluster: cd {GALLERY_DIR} && python3 -m http.server 8080")
print(f"  On local:   ssh -L 8080:localhost:8080 vsc21215@login1-vaughan.hpc.uantwerpen.be")
print(f"  Browser:    http://localhost:8080/gallery.html")
