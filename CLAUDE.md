# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of Ultralytics YOLO (v8.4.2) with custom **contour-based attention mechanisms** (CBAM) for a thesis project. The key modification adds OpenCV-based contour detection as a spatial attention hint within YOLO detection heads, targeting improved object detection (e.g., glaucoma detection in `GlaucomaYOLO_resized/` dataset).

## Common Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_python.py

# Run a specific test
pytest tests/test_python.py::test_predict

# Run with verbose output and no doctest
pytest tests/test_python.py -v --override-ini="addopts="

# Lint with ruff (line length 120)
ruff check ultralytics/
ruff format ultralytics/

# CLI usage
yolo predict model=yolo11n.pt source=image.jpg
yolo train model=yolo11n.pt data=coco8.yaml epochs=100
yolo val model=yolo11n.pt data=coco8.yaml
```

## Architecture

### Engine Layer (`ultralytics/engine/`)
Core training/inference pipeline. All model types share these base classes:
- `model.py` — Base `Model` class; entry point for `YOLO(...)` API
- `trainer.py` — Training loop (mixed precision, multi-GPU, callbacks)
- `validator.py` — Validation with metrics
- `predictor.py` — Inference pipeline
- `exporter.py` — Export to ONNX, TensorFlow, OpenVINO, CoreML, etc.

### Model Definitions (`ultralytics/models/yolo/`)
`model.py` contains the `YOLO` class which auto-selects task-specific trainer/validator/predictor via `task_map`. Each task (detect, segment, classify, pose, obb) has its own subdirectory with train/predict/val implementations.

### Neural Network Modules (`ultralytics/nn/`)
- `tasks.py` — Model construction from YAML configs; `parse_model()` builds the PyTorch model from YAML layer definitions
- `modules/block.py` — Building blocks (C3k2, SPPF, C2PSA, etc.) — **also contains the custom CBAM/ChannelAttention/SpatialAttention classes**
- `modules/conv.py` — Convolution variants (Conv, DWConv, etc.) — has an upstream CBAM class that is overridden by the one in block.py
- `modules/head.py` — Detection/segmentation/pose heads

### Model YAML Configs (`ultralytics/cfg/models/`)
Model architectures are defined as YAML files. The custom CBAM configs are:
- `11/yolo11n-cbam.yaml` — nano variant
- `11/yolo11s-cbam.yaml` — small variant
- `11/yolo11m-cbam.yaml` — medium variant

These insert `CBAM` modules after each P3/P4/P5 output in the detection head before the final `Detect` layer.

## Custom Attention Mechanism (Key Modification)

The contour-based attention is implemented in two locations:
1. **`modified_yolo.py`** (standalone version) — reference implementation with all classes
2. **`ultralytics/nn/modules/block.py`** (integrated) — production version used by the framework

Three modules work together:
- **`ChannelAttention`** — AdaptiveAvgPool + AdaptiveMaxPool → MLP → sigmoid. Weights initialized to zero so output starts at 0.5.
- **`SpatialAttention`** — Converts feature maps to grayscale → GaussianBlur → Canny edge detection → `cv2.findContours` → filled contour mask. Multi-scale conv branches (kernels 1,3,5) refine the mask. Initialized to output zero (identity behavior).
- **`CBAM`** — Combines both: additive channel attention residual (`x + x * ca(x)`) followed by spatial attention. Designed for stable training from initialization.

**Important**: The contour computation (`compute_contours`) runs on CPU using OpenCV (detached from the computation graph). This is intentional for the thesis but impacts training speed.

### How CBAM integrates with model parsing
In `ultralytics/nn/tasks.py`, `parse_model()` has a special case (line ~1622): when `m is CBAM`, it sets `c1 = c2 = ch[f]` (channel passthrough) and passes them as args. The CBAM import comes from `ultralytics.nn.modules.block` (not `conv`), so the contour-based version is used.

## Code Style
- Line length: 120 characters
- Linting: ruff
- Formatting: ruff format, yapf (PEP8-based)
- Docstrings: Google style, formatted with docformatter
- Import sorting: isort (line_length=120)
