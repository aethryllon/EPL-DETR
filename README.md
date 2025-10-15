<!--
EPL-DETR Minimal Release (English-only)
Generated for reviewer-facing submission package. Contains a compact set of files needed to build and inspect the model architecture.
-->

<div align="center">
  <h1>EPL-DETR</h1>
  <h3>Enhanced Industrial PPE Detection: A Laplacian-Refined Transformer with Polarized Attention</h3>


  <p><strong>Authors:</strong> (author list withheld — manuscript not yet submitted)</p>

  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Abstract

Ensuring correct use of Personal Protective Equipment (PPE) in industrial environments is critical for workplace safety, yet automated PPE detection remains challenging due to sensor noise, illumination variations, occlusion, and complex backgrounds. This paper presents EPL-DETR, an Edge-Refined Laplacian Detection Transformer with Polarized Attention and Layered Dense–Sparse features, tailored for robust industrial PPE recognition. EPL-DETR integrates a Laplacian-of-Gaussian based LoGStem for edge-aware, noise-robust feature extraction, a polarized linear attention mechanism to improve discriminative feature interaction with linear complexity, and a Layered Dense–Sparse (LDS) module for efficient multi-granular context aggregation. Experiments on the SH17 (8,099 images, 17 classes) and SFCHD (12,373 images, 7 classes) datasets show that EPL-DETR-M achieves 64.3% mAP@0.5 and 78.2% mAP@0.5, respectively, with 18.8M parameters, outperforming several state-of-the-art detectors while maintaining favorable efficiency. Knowledge distillation further yields a compact 9.1M-parameter variant achieving 61.3% mAP@0.5. These results demonstrate that domain-oriented architectural refinements can substantially enhance detection accuracy and robustness for safety-critical industrial applications.

---

## Repository overview

- `EPL-DETR.yaml` — model configuration file defining the EPL-DETR architecture.
- `train.py` — main training script for the EPL-DETR model.
- `val.py` — validation script for evaluating model performance.
- `detect.py` — inference script for running object detection on images.
- `dataset/` — directory structure for organizing training and validation data.
- `weights/` — directory for storing trained model weights.
- `ultralytics/` — core framework files based on the Ultralytics YOLO implementation.

> Note: This repository contains the implementation of EPL-DETR, built upon the Ultralytics framework. The ultralytics/ directory contains modified and extended components to support the EPL-DETR architecture.

## Approach

EPL-DETR combines three domain-oriented architectural components:

- LoGStem: a Laplacian-of-Gaussian based stem that enhances edge-aware, noise-robust low-level features.
- Polarized Linear Attention: a linear-complexity attention block that separates polarity-aware projection and improves discriminative interactions.
- Layered Dense–Sparse (LDS) Module: efficient multi-granular context aggregation using alternating dense and sparse connections to capture both local detail and global context.

The resulting architecture balances accuracy and efficiency for industrial PPE detection tasks.

## Installation

### Option 1 — Conda (recommended)

```powershell
conda create -n epldetr python=3.10 -y
conda activate epldetr
pip install -r requirements_infer.txt
```

### Option 2 — venv

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements_infer.txt
```

If you need a CUDA-enabled PyTorch wheel, follow the official PyTorch installation guide: https://pytorch.org/get-started/locally/

## Dataset

To use your own dataset with EPL-DETR, organize your data in the following structure:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image should have a corresponding label file in YOLO format (.txt) with the same filename. The label files should contain annotations in the format: `class_id center_x center_y width height` (normalized to 0-1).

Create a `data.yaml` file in the dataset directory with the following content:

```yaml
train: dataset/images/train
val: dataset/images/val

nc: [number of classes]
names: [list of class names]
```

Replace `[number of classes]` with the actual number of classes in your dataset and `[list of class names]` with the actual class names.

## Training & Evaluation

1) Training:

```bash
python train.py --config EPL-DETR.yaml --data dataset/data.yaml --epochs 100
```

2) Validation:

```bash
python val.py --model weights/best.pt --data dataset/data.yaml
```

3) Inference:

```bash
python detect.py --model weights/best.pt --image path/to/image.jpg --conf 0.5
```

Make sure to adjust the paths and parameters according to your setup. The trained weights will be saved in the `weights/` directory.

## Main results (paper highlights)

EPL-DETR demonstrates strong performance on the two industrial PPE benchmarks reported in the manuscript:

### Performance comparison on the SH17 dataset. Best results are in **bold**.

| Model | P(%) | R(%) | mAP@0.5(%) | mAP@0.5:0.95(%) | Params(M) | GFLOPS |
|-------|------|------|-------------|------------------|-----------|--------|
| RT-DETR-r18 | 66.5 | 55.0 | 57.8 | 38.2 | 19.9 | 57.0 |
| YOLOv5m | 63.7 | 57.8 | 59.0 | 38.1 | 21.2 | 49.0 |
| YOLOv6m | 64.9 | 50.0 | 52.8 | 32.9 | 34.9 | 85.8 |
| YOLOv8m | 71.4 | 54.1 | 58.7 | 37.8 | 25.9 | 78.9 |
| YOLOv8m-worldv2 | 67.4 | 54.1 | 57.1 | 37.0 | 28.4 | 90.3 |
| YOLOv8m-rtdetr | 66.2 | 58.2 | 59.2 | 38.9 | 26.1 | 70.2 |
| YOLOv9m | 73.8 | 53.9 | 62.2 | 40.2 | 20.1 | 76.8 |
| YOLOv10m | 63.5 | 52.0 | 56.4 | 36.5 | **15.4** | 59.1 |
| YOLO11m | 68.4 | 57.3 | 60.5 | 39.1 | 20.1 | 68.0 |
| YOLO12m | 68.5 | 53.4 | 57.0 | 36.8 | 20.2 | 67.5 |
| Mamba-YOLO-B | 72.0 | 53.2 | 58.4 | 37.8 | 19.1 | **45.4** |
| Hyper-YOLO-M | **77.1** | 54.2 | 60.6 | 39.5 | 33.3 | 103.3 |
| DMS-DETR-r18 | 71.3 | 54.9 | 59.3 | 39.0 | 27.3 | 53.6 |
| EPL-DETR-M | 75.9 | **60.6** | **64.3** | **41.1** | 18.8 | 61.5 |

### Performance comparison on the SFCHD dataset. Best results are in **bold**.

| Model | P(%) | R(%) | mAP@0.5(%) | mAP@0.5:0.95(%) | Params(M) | GFLOPS |
|-------|------|------|-------------|------------------|-----------|--------|
| RT-DETR-r18 | 82.4 | 76.8 | 76.7 | 50.6 | 19.9 | 57.0 |
| YOLOv5m | 78.3 | 72.4 | 75.8 | 51.0 | 21.2 | 49.0 |
| YOLOv6m | 77.3 | 70.2 | 73.7 | 40.6 | 34.9 | 85.8 |
| YOLOv8m | 79.4 | 72.4 | 76.5 | 51.5 | 25.9 | 78.9 |
| YOLOv8m-worldv2 | 80.2 | 71.5 | 76.6 | 51.9 | 28.4 | 90.3 |
| YOLOv8m-detr | **83.6** | 77.6 | 77.6 | 51.2 | 26.1 | 70.2 |
| YOLOv9m | 81.4 | 72.3 | 76.7 | 51.8 | 20.1 | 76.8 |
| YOLOv10m | 80.1 | 71.0 | 76.0 | 50.9 | **15.4** | 59.1 |
| YOLO11m | 78.4 | 74.1 | 76.9 | 51.9 | 20.1 | 68.0 |
| YOLO12m | 79.5 | 72.9 | 76.6 | 51.7 | 20.2 | 67.5 |
| Mamba-YOLO-B | 79.7 | 72.3 | 76.4 | 52.2 | 19.1 | **45.4** |
| Hyper-YOLO-M | 79.8 | 73.2 | 76.9 | **52.5** | 33.3 | 103.3 |
| EPL-DETR-M | 82.9 | **79.4** | **78.2** | 52.3 | 18.8 | 61.5 |

## Visualization and qualitative results

Example qualitative outputs (place your figures under `outputs/` and reference them here):

Include your visualization files in the repository (or attach in the submission) for reviewer inspection.

## License

This minimal release contains original code added for the submission (you may choose an appropriate license, e.g., MIT or Apache-2.0) and also bundles third-party implementation files retained for reproducibility. Bundled third-party files retain their original license (AGPL-3.0) and must not have their license notices removed.

## Acknowledgements

This work builds on top of existing open-source detection frameworks and the community implementations that made this research possible. Please see the `ultralytics/` folder for original copyright and license attributions.

## Contact

When you are ready, paste the final author list and preferred citation (DOI or arXiv id) and I will insert a BibTeX entry and update the header accordingly. I can also prepare a strict minimal ZIP containing only reviewer-facing files if you want to upload directly to GitHub.


Add the exact DOI / arXiv link and the author list and I will update this section.

