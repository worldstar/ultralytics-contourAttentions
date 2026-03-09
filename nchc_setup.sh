#!/bin/bash
# ============================================================================
# NCHC Quick Setup Script
# ============================================================================
# Run this ONCE before submitting the Slurm job array.
#
# Usage:
#   bash nchc_setup.sh
# ============================================================================

PROJECT_DIR="$HOME/ultralytics-contourAttentions"

echo "=== NCHC Project Setup ==="
echo "Project dir: $PROJECT_DIR"

# 1. Create logs directory
mkdir -p "$PROJECT_DIR/logs"
echo "[OK] Created logs directory"

# 2. Verify dataset exists
if [ -d "$PROJECT_DIR/GlaucomaYOLO_resized" ]; then
    TRAIN_COUNT=$(ls "$PROJECT_DIR/GlaucomaYOLO_resized/images/train/" 2>/dev/null | wc -l)
    VAL_COUNT=$(ls "$PROJECT_DIR/GlaucomaYOLO_resized/images/val/" 2>/dev/null | wc -l)
    TEST_COUNT=$(ls "$PROJECT_DIR/GlaucomaYOLO_resized/images/test/" 2>/dev/null | wc -l)
    echo "[OK] Dataset found: train=$TRAIN_COUNT, val=$VAL_COUNT, test=$TEST_COUNT"
else
    echo "[ERROR] Dataset not found at $PROJECT_DIR/GlaucomaYOLO_resized/"
    echo "        Please upload the dataset before running experiments."
    exit 1
fi

# 3. Verify all YAML configs exist
CONFIGS=(
    "ultralytics/cfg/models/v9/yolov9c.yaml"
    "ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml"
    "ultralytics/cfg/models/v9/yolov9c-stdcbam.yaml"
    "ultralytics/cfg/models/v9/yolov9c-se.yaml"
    "ultralytics/cfg/models/v9/yolov9c-eca.yaml"
    "ultralytics/cfg/models/v9/yolov9c-simam.yaml"
    "ultralytics/cfg/models/v9/yolov9c-coordatt.yaml"
    "ultralytics/cfg/models/v9/yolov9c-gam.yaml"
)
ALL_OK=true
for cfg in "${CONFIGS[@]}"; do
    if [ -f "$PROJECT_DIR/$cfg" ]; then
        echo "[OK] $cfg"
    else
        echo "[MISSING] $cfg"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo "[ERROR] Some config files are missing!"
    exit 1
fi

# 4. Verify Singularity container
CONTAINER="/work/hpc_sys/sifs/pytorch_23.11-py3.sif"
if [ -f "$CONTAINER" ]; then
    echo "[OK] Singularity container found"
else
    echo "[WARN] Container not found at $CONTAINER"
    echo "       Check the correct path on your NCHC partition"
fi

# 5. Verify glaucoma.yaml data path is relative
echo ""
echo "=== glaucoma.yaml content ==="
cat "$PROJECT_DIR/GlaucomaYOLO_resized/glaucoma.yaml"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To submit all 40 experiments:"
echo "  cd $PROJECT_DIR"
echo "  sbatch nchc_experiments.slurm"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To check a specific job's output:"
echo "  cat logs/exp_<JOB_ID>_<TASK_ID>.out"
echo ""
echo "After all jobs complete, collect results:"
echo "  python nchc_collect_results.py"
