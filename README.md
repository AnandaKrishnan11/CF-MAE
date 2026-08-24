## CF-MAE: A Change-Fused Masked Autoencoder for Earth Observation-Based Multi-Hazard Change Detection
> [Ananda Krishnan](https://scholar.google.com/citations?user=YWZ3kKsAAAAJ&hl=en), Getachew Workineh Gella, Bruno Menini Matosak, Stefan Lang

>Official PyTorch implementation of CF-MAE, a change-aware self-supervised pre-training framework for multi-hazard change detection from bi-temporal remote sensing imagery.

> Presented at **[IGARSS 2026](https://2026.ieeeigarss.org/)** — Washington, D.C

---

## Overview

CF-MAE addresses change detection by extending masked autoencoding into a bi-temporal setting, learning change-aware representations directly during pre-training.

For the downstream task, we introduce the Change-Fused Transformer (CFT) in three variants:

- **CFT v1 (S-CFT)** — single-scale, using change features directly from the fusion module
- **CFT v2 (MS-CFT)** — multi-scale, fusing change features across encoder stages with skip connections
- **CFT v3** — multi-scale, fusing change features across encoder stages directly to decoder after concatenation.
---

## Pre-training Architecture

<p align="center">
  <img src="assets/CF_MAE.png" width="750" alt="CF-MAE architecture overview">
</p>
See assets for the architecture diagrams of each CFT variant.

---

## Dataset

Pre-training uses **PANDORA** (Paired Natural Disaster Observations for Reconstruction and Analysis), a multi-source, multi-modal dataset compiled from openly available benchmarks. It contains 376,634 paired pre-/post-disaster scenes, chipped to 3 × 224 × 224. 

| Dataset            | Images  | Modality |
|--------------------|---------|----------|
| AI4EO (ESA Challenge) | 20,426  | Optical  |
| LEVIR              | 10,192  | Optical  |
| Wuhan              | 6,393   | Optical  |
| xBD                | 176,544 | Optical  |
| SenForFlood        | 163,079 | SAR      |

Downstream fine-tuning and evaluation use:

- **S2-WCD** — Sentinel-2 Wildfire Change Detection Dataset
- **DSIFN-CD** — very high resolution building change detection
- **Maui** — fire damage assessment (ESRI World Wayback pre-image, Vantor open data post-image)

Access Data via: Will be released soon

---

## Dataset Structure

### Pre-training

```
<PATH>/
├── ai4eo/
│   ├── pre/
│   │   ├── 000000.tif
│   │   ├── 000001.tif
│   │   └── ...
│   └── post/
│       ├── 000000.tif
│       ├── 000001.tif
│       └── ...
├── levir/
│   ├── pre/
│   └── post/
├── wuhan/
│   ├── pre/
│   └── post/
├── xbd/
│   ├── pre/
│   └── post/
└── senforflood/
    ├── pre/
    └── post/
```

### Fine-tuning

`PATH` points to a directory containing `train/`, `val/` and `test/` splits, each with `pre/`, `post/` and `label/` directories.

```
<PATH>/
├── train/
│   ├── pre/
│   │   ├── pre_0.tif
│   │   ├── pre_1.tif
│   │   └── ...
│   ├── post/
│   │   ├── post_0.tif
│   │   ├── post_1.tif
│   │   └── ...
│   └── label/
│       ├── label_0.tif
│       ├── label_1.tif
│       └── ...
├── val/
│   ├── pre/
│   ├── post/
│   └── label/
└── test/
    ├── pre/
    ├── post/
    └── label/
```

## Results

Quantitative evaluation (F1 & IoU of change class) against CNN, transformer, and foundation-model baselines:

| Model | S2-WCD F1 | S2-WCD IoU | DSIFN-CD F1 | DSIFN-CD IoU | Maui F1 | Maui IoU |
|--------------------|:---------:|:----------:|:-----------:|:------------:|:-------:|:--------:|
| FC-EF              | 0.83 | 0.71 | 0.53 | 0.36 | 0.75 | 0.60 |
| FC-Siam-diff       | 0.59 | 0.42 | 0.34 | 0.21 | 0.74 | 0.59 |
| FC-Siam-conc       | 0.83 | 0.72 | 0.58 | 0.41 | 0.74 | 0.58 |
| DTCDSCN            | 0.89 | 0.80 | 0.70 | 0.54 | 0.74 | 0.60 |
| BIT                | 0.90 | 0.82 | 0.73 | 0.57 | 0.75 | 0.59 |
| ChangeFormer       | 0.89 | 0.80 | 0.62 | 0.45 | 0.60 | 0.43 |
| SatMAE             | 0.73 | 0.36 | 0.48 | 0.27 | 0.54 | 0.34 |
| CLAY               | 0.78 | 0.71 | 0.60 | 0.38 | 0.76 | 0.68 |
| Prithvi            | 0.86 | 0.73 | 0.66 | 0.40 | 0.64 | 0.55 |
| **CFTv2 (ImageNet)** | 0.93 | **0.87** | 0.90 | 0.74 | 0.81 | 0.73 |
| **CFTv2 (SatMAE)**   | 0.78 | 0.71 | 0.81 | 0.61 | 0.69 | 0.60 |
| **CFTv1 (PANDORA)**  | 0.91 | 0.83 | 0.90 | 0.75 | 0.81 | **0.74** |
| **CFTv2 (PANDORA)**  | **0.94** | 0.84 | 0.93 | 0.81 | **0.82** | **0.74** |
| **CFTv3 (PANDORA)**  | **0.94** | 0.84 | **0.94** | **0.82** | **0.82** | **0.74** |
---

## Installation

```bash
git clone https://github.com/AnandaKrishnan11/CF-MAE.git
cd CF-MAE

# Create environment
conda create -n cfmae python=3.10 -y
conda activate cfmae

# Install dependencies
pip install -r requirements.txt
```

---
## Weights
### Pre-trained
PANDORA: https://github.com/AnandaKrishnan11/CF-MAE/releases/tag/v1.0-pandora

### Finetuned (CFTv2)
Will be released soon

## Usage

### Pre-training

```bash
python main_pretrain.py --data-path --output --lr --epochs 
```

### Fine-tuning (downstream change detection)

```bash
python main_finetune.py --data-path --output --version --pretrained --lr --epochs
```

### Evaluation

```bash
python inference.py --data-path --output --checkpoint --version
```

> Update dataset paths and hyperparameters in the corresponding config files under `configs/`.

---

## Training Details

- **Pre-training:** 100 epochs, 75% masking ratio, MSE loss, AdamW, base LR 1e-4 with 10-epoch warm-up and cosine decay, effective batch size 1,024 (via gradient accumulation).
- **Fine-tuning:** 50 epochs, combined cross-entropy + Dice loss, AdamW with cosine schedule and layer-wise learning rates.
- **Hardware:** NVIDIA RTX 3090 GPU.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{krishnan2026cfmae,
  title     = {CF-MAE: A Change-Fused Masked Autoencoder for Earth Observation-Based Multi-Hazard Change Detection},
  author    = {Krishnan, Ananda and Gella, Getachew Workineh and Matosak, Bruno Menini and Lang, Stefan},
  booktitle = {IGARSS 2026 - IEEE International Geoscience and Remote Sensing Symposium},
  year      = {2026},
  organization = {IEEE}
}
```

