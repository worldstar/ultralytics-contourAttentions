#!/bin/bash
#SBATCH --job-name=yolov9_attn
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=MSTXXXXX
#SBATCH --array=0-39
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

# ============================================================================
# NCHC H100 Slurm Job Array for 40 Experiments (8 configs x 5 seeds)
# ============================================================================
#
# Usage:
#   sbatch nchc_experiments.slurm
#
# To run a subset (e.g., jobs 0-9 only):
#   sbatch --array=0-9 nchc_experiments.slurm
#
# To rerun a single failed job (e.g., job 17):
#   sbatch --array=17 nchc_experiments.slurm
#
# Job mapping (SLURM_ARRAY_TASK_ID):
#   0-4:   baseline      seeds 0-4
#   5-9:   contourcbam   seeds 0-4
#   10-14: stdcbam       seeds 0-4
#   15-19: se            seeds 0-4
#   20-24: eca           seeds 0-4
#   25-29: simam         seeds 0-4
#   30-34: coordatt      seeds 0-4
#   35-39: gam           seeds 0-4
# ============================================================================

CONTAINER="/work/hpc_sys/sifs/pytorch_23.11-py3.sif"
PROJECT_DIR="$HOME/ultralytics-contourAttentions"

# --- Experiment configuration ---
CONFIGS=(
    "baseline:ultralytics/cfg/models/v9/yolov9c.yaml"
    "contourcbam:ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml"
    "stdcbam:ultralytics/cfg/models/v9/yolov9c-stdcbam.yaml"
    "se:ultralytics/cfg/models/v9/yolov9c-se.yaml"
    "eca:ultralytics/cfg/models/v9/yolov9c-eca.yaml"
    "simam:ultralytics/cfg/models/v9/yolov9c-simam.yaml"
    "coordatt:ultralytics/cfg/models/v9/yolov9c-coordatt.yaml"
    "gam:ultralytics/cfg/models/v9/yolov9c-gam.yaml"
)
SEEDS=(0 1 2 3 4)

# --- Derive experiment name and seed from array task ID ---
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

CONFIG_ENTRY="${CONFIGS[$CONFIG_IDX]}"
EXP_NAME="${CONFIG_ENTRY%%:*}"
MODEL_YAML="${CONFIG_ENTRY##*:}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="${EXP_NAME}_seed${SEED}"

echo "============================================================"
echo "  Job Array ID:  $SLURM_ARRAY_TASK_ID"
echo "  Experiment:    $EXP_NAME"
echo "  Model YAML:    $MODEL_YAML"
echo "  Seed:          $SEED"
echo "  Run Name:      $RUN_NAME"
echo "  Start Time:    $(date)"
echo "============================================================"

# --- Setup ---
module purge
module load singularity
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Training hyperparameters (matching the paper) ---
BATCH=32
EPOCHS=100
IMGSZ=640
OPTIMIZER="SGD"
CLOSE_MOSAIC=15
WORKERS=8
DATA_YAML="GlaucomaYOLO_resized/glaucoma.yaml"

singularity exec --nv --bind "$HOME:$HOME" "$CONTAINER" bash -lc "
    cd $PROJECT_DIR

    # Install dependencies
    python -m pip install --user --upgrade pip
    python -m pip install --user opencv-python-headless>=4.6.0
    python -m pip install --user seaborn scipy matplotlib pandas pyyaml tqdm pillow psutil py-cpuinfo thop

    echo '--- Environment check ---'
    python -c 'import cv2; print(\"OpenCV version:\", cv2.__version__)'
    python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available(), \"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")'
    echo '--- Starting training ---'

    # Training
    python -u -c \"
import sys
sys.path.insert(0, '.')
from ultralytics import YOLO

model = YOLO('${MODEL_YAML}')
results = model.train(
    data='${DATA_YAML}',
    epochs=${EPOCHS},
    batch=${BATCH},
    imgsz=${IMGSZ},
    device=0,
    workers=${WORKERS},
    optimizer='${OPTIMIZER}',
    close_mosaic=${CLOSE_MOSAIC},
    seed=${SEED},
    name='${RUN_NAME}',
    exist_ok=True,
    verbose=True,
    val=True,
    plots=True,
)
print('Training complete for ${RUN_NAME}')

# Run test-set evaluation with best weights
from pathlib import Path
best_weights = Path('runs/detect/${RUN_NAME}/weights/best.pt')
if best_weights.exists():
    val_model = YOLO(str(best_weights))
    val_results = val_model.val(
        data='${DATA_YAML}',
        split='test',
        imgsz=${IMGSZ},
        batch=${BATCH},
        device=0,
        workers=${WORKERS},
        name='${RUN_NAME}_test',
        exist_ok=True,
        verbose=True,
    )
    box = val_results.box
    print(f'Test mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}  P={box.mp:.4f}  R={box.mr:.4f}')
    for i, cls_name in val_results.names.items():
        if i < len(box.ap50):
            print(f'  {cls_name}: AP50={box.ap50[i]:.4f}  AP50-95={box.ap[i]:.4f}')
else:
    print('WARNING: best.pt not found, skipping test evaluation')
\"

    echo '--- Job finished at $(date) ---'
"
