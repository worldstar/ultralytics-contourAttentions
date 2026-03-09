# YOLOv9 + ContourCBAM for Glaucoma Detection (Ultralytics Framework)

A plug-and-play **Contour-Based Attention Module (ContourCBAM)** integrated into [YOLOv9](https://github.com/WongKinYiu/yolov9) via the [Ultralytics](https://github.com/ultralytics/ultralytics) framework for detecting the Optic Disc and Optic Cup in retinal fundus images. The module injects explicit geometric priors from Canny edge detection into the detection head, improving localization of the faint Optic Cup boundary — critical for accurate Cup-to-Disc Ratio (CDR) measurement in automated glaucoma screening.

Based on the paper: [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

## ContourCBAM Module

The ContourCBAM replaces the learned spatial attention in standard CBAM with a **non-differentiable contour-based structural hint** refined through learnable convolutional branches. It is composed of three components defined in `ultralytics/nn/modules/block.py`:

| Component | Class Name | Description |
|-----------|-----------|-------------|
| Channel Attention | `ChannelAttention` | Dual-pathway (avg-pool + max-pool) squeeze-excitation with shared MLP |
| Spatial Attention | `SpatialAttention` | Canny edge detection + `cv2.findContours` on feature maps, refined by parallel multi-scale conv branches (1x1, 3x3, 5x5) |
| Combined Module | `CBAM` | Applies channel then spatial attention with **additive residual connections** |

### How the Spatial Attention Works

1. Feature maps are detached from the autograd graph and converted to grayscale on CPU
2. Gaussian blur (5x5) smooths high-frequency noise
3. **Canny edge detection** with auto-thresholding extracts structural boundaries
4. `cv2.findContours` + `cv2.drawContours` produces filled contour masks
5. The contour mask is refined through three parallel Conv+BN branches (kernel sizes 1, 3, 5)
6. A sigmoid activation produces the spatial attention map
7. Attention is applied via **additive residual**: `output = x * mask + x`

### Stable Initialization

Both attention components use zero-initialization on their final layers:

- **Channel Attention**: Last MLP conv weights = 0, so initial output = sigmoid(0) = 0.5
- **Spatial Attention**: All conv branch weights = 0, so initial mask = sigmoid(0) = 0.5

Combined with the additive residual design, the module acts as an approximate **identity function at initialization**, ensuring stable gradient flow from the start of training.

## Architecture

The ContourCBAM is placed in the YOLOv9-C detection neck (configured in `ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml`) after the `RepNCSPELAN4` feature processing blocks at all three pyramid levels, just before the `Detect` head:

```
 BACKBONE                                    HEAD
 ┌────────────────────────────────────┐ ┌──────────────────────────────────────────────────┐
 │                                    │ │                                                  │
 │  Input                             │ │  SPPELAN ──► Upsample+Concat ──► RepNCSPELAN4    │
 │    │                               │ │                  │                                │
 │    ▼                               │ │       Upsample+Concat ──► RepNCSPELAN4           │
 │  Conv ──► Conv ──► RepNCSPELAN4    │ │                  │                                │
 │                                    │ │           ┌──► CBAM (P3) ─────┐                   │
 │  ADown                             │ │           │                   │                   │
 │    │                               │ │  ADown+Concat ──► RepNCSPELAN4                   │
 │  RepNCSPELAN4 ────┼───┤            │ │                   │                               │
 │                   │   │            │ │           ┌──► CBAM (P4) ─────┤                   │
 │  ADown            │   │            │ │           │                   │                   │
 │    │              │   │            │ │  ADown+Concat ──► RepNCSPELAN4                   │
 │  RepNCSPELAN4 ────┼───┤            │ │                   │                               │
 │                   │   │            │ │           ┌──► CBAM (P5) ─────┤                   │
 │  ADown            │   │            │ │           │                   │                   │
 │    │              │   │            │ │           │                   │                   │
 │  RepNCSPELAN4 ────┼───┘            │ │  Detect  ◄────────────────────┘                   │
 │                                    │ │           (P3, P4, P5)                            │
 └────────────────────────────────────┘ └──────────────────────────────────────────────────┘
```

The CBAM placement in `yolov9c-contourcbam.yaml`:

- **Layer 16** — CBAM after P3 features (256 channels, 8x downsample)
- **Layer 20** — CBAM after P4 features (512 channels, 16x downsample)
- **Layer 24** — CBAM after P5 features (512 channels, 32x downsample)
- **Layer 25** — `Detect` head receives outputs from layers `[16, 20, 24]`

## Glaucoma Detection Dataset

This repository includes the `GlaucomaYOLO_resized/` dataset (combined ORIGA + G1020) for optic disc and optic cup detection.

### Dataset Structure

```
GlaucomaYOLO_resized/
├── glaucoma.yaml          # Dataset configuration file
├── images/
│   ├── train/             # 998 training images
│   ├── val/               # 109 validation images
│   └── test/              # 227 test images
└── labels/
    ├── train/             # YOLO-format labels (class x_center y_center width height)
    ├── val/
    └── test/
```

**Classes (2):**

| ID | Class       |
|----|-------------|
| 0  | optic_disc  |
| 1  | optic_cup   |

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch >= 1.8, OpenCV >= 4.6.0, NumPy, SciPy, seaborn.

## Training

```bash
yolo train model=ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml \
     data=GlaucomaYOLO_resized/glaucoma.yaml \
     epochs=100 batch=32 imgsz=640 optimizer=SGD close_mosaic=15
```

Or via Python:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml")
model.train(
    data="GlaucomaYOLO_resized/glaucoma.yaml",
    epochs=100, batch=32, imgsz=640,
    optimizer="SGD", close_mosaic=15,
)
```

## Evaluation

```bash
yolo val model=runs/detect/contourcbam_seed0/weights/best.pt \
     data=GlaucomaYOLO_resized/glaucoma.yaml split=test imgsz=640
```

## Inference

```bash
yolo predict model=runs/detect/contourcbam_seed0/weights/best.pt \
     source='path/to/fundus/images/' imgsz=640
```

## Results

Validated over **5 runs** on **1,334 annotated fundus images** with two classes (Optic Disc, Optic Cup):

| Metric | Baseline (YOLOv9) | + ContourCBAM (P3+P4+P5) |
|--------|-------------------|--------------------------|
| Optic Cup mAP50 | 0.975 | **0.988** (+1.33%) |
| Overall mAP50 | — | **0.992** |
| Cup False Positives | baseline | **reduced by 50%** |

**Ablation**: Multi-scale placement (P3+P4+P5) was confirmed optimal compared to single-scale variants.

## Cross-Architecture Applicability

The ContourCBAM has also been integrated into **YOLOv12** variants, demonstrating its generalizability across YOLO architectures. The YOLOv12 implementation is available in a separate repository.

## NCHC H100 Deployment (40 Experiments)

This section describes how to run the full set of 40 experiments (8 attention configs x 5 seeds) on the NCHC (National Center for High-performance Computing) Taiwan Computing Cloud using Slurm and Singularity.

### Experiment Configurations

| Config | Model YAML | Attention Module | Params |
|--------|-----------|------------------|--------|
| baseline | `yolov9c.yaml` | None | 25,590,912 |
| contourcbam | `yolov9c-contourcbam.yaml` | ContourCBAM (proposed) | 25,664,763 |
| stdcbam | `yolov9c-stdcbam.yaml` | Standard CBAM (Woo et al. 2018) | 25,664,934 |
| se | `yolov9c-se.yaml` | SE (Hu et al. 2018) | 25,664,640 |
| eca | `yolov9c-eca.yaml` | ECA (Wang et al. 2020) | 25,590,927 |
| simam | `yolov9c-simam.yaml` | SimAM (Yang et al. 2021) | 25,590,912 |
| coordatt | `yolov9c-coordatt.yaml` | Coordinate Attention (Hou et al. 2021) | 25,646,288 |
| gam | `yolov9c-gam.yaml` | GAM (Liu et al. 2021) | 40,339,712 |

Each config is trained with 5 seeds (0-4), totaling **40 experiments**.

Training hyperparameters: `batch=32, epochs=100, optimizer=SGD, close_mosaic=15, imgsz=640`.

### Slurm Job Array Mapping

The `nchc_experiments.slurm` file uses `--array=0-39` where each task ID maps to:

| Task IDs | Config | Seeds |
|----------|--------|-------|
| 0-4 | baseline | 0-4 |
| 5-9 | contourcbam | 0-4 |
| 10-14 | stdcbam | 0-4 |
| 15-19 | se | 0-4 |
| 20-24 | eca | 0-4 |
| 25-29 | simam | 0-4 |
| 30-34 | coordatt | 0-4 |
| 35-39 | gam | 0-4 |

### Deployment Steps

1. **Upload the repository** to `$HOME/ultralytics-contourAttentions` on NCHC.

2. **Edit the Slurm account**: Open `nchc_experiments.slurm` and replace `MSTXXXXX` on line 7 with your actual NCHC account code.

3. **Run the setup check**:
   ```bash
   cd $HOME/ultralytics-contourAttentions
   bash nchc_setup.sh
   ```
   This verifies the dataset, all 8 YAML configs, and the Singularity container are in place.

4. **Submit all 40 experiments**:
   ```bash
   sbatch nchc_experiments.slurm
   ```
   To submit only a subset (e.g., baseline seeds 0-4):
   ```bash
   sbatch --array=0-4 nchc_experiments.slurm
   ```
   To rerun a single failed job (e.g., task ID 17):
   ```bash
   sbatch --array=17 nchc_experiments.slurm
   ```

5. **Monitor job status**:
   ```bash
   squeue -u $USER
   ```
   Check a specific job's output:
   ```bash
   cat logs/exp_<JOB_ID>_<TASK_ID>.out
   ```

6. **Collect results** after all jobs complete:
   ```bash
   python nchc_collect_results.py
   ```
   This generates `experiment_results/all_results.json`, `experiment_results/summary.json`, and prints mean +/- std tables with paired t-tests (baseline vs. each attention module).

## Key Files

| File | Purpose |
|------|---------|
| `ultralytics/nn/modules/block.py` | ContourCBAM module classes (`ChannelAttention`, `SpatialAttention`, `CBAM`) and all alternative attention modules |
| `ultralytics/nn/tasks.py` | `parse_model()` with attention module registration |
| `ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml` | YOLOv9-C architecture config with ContourCBAM at P3/P4/P5 |
| `ultralytics/cfg/models/v9/yolov9c.yaml` | Baseline YOLOv9-C without attention |
| `GlaucomaYOLO_resized/glaucoma.yaml` | Dataset configuration (2 classes, 1334 images) |
| `nchc_experiments.slurm` | Slurm job array script (40 jobs, H100 GPU, Singularity container) |
| `nchc_setup.sh` | Pre-flight setup check (dataset, configs, container) |
| `nchc_collect_results.py` | Post-experiment result aggregation and statistical summary |
| `run_all_experiments.py` | Alternative local experiment runner with auto-resume |

## Citation

If you use the ContourCBAM module in your work, please cite:

```
<!-- Add your paper citation here -->
```

This work builds on YOLOv9:

```
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}
```

## Acknowledgements

- [YOLOv9 — WongKinYiu](https://github.com/WongKinYiu/yolov9)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [YOLOR](https://github.com/WongKinYiu/yolor)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
