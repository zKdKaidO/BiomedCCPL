# BiomedCCPL: Causal Conditional Prompt Learning for Biomedical Vision-Language Models

[[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Cui_BiomedCCPL_Causal_Conditional_Prompt_Learning_for_Biomedical_Vision-Language_Models_CVPR_2026_paper.html)]

Xueliang Cui, Juncai Zhang, Jiacheng Hou, Dan Lu, Hao Zhang, and Ruxin Wang

## Overview

BiomedCCPL adapts a frozen BiomedCLIP backbone to few-shot biomedical image classification while improving generalization from seen to unseen classes. It contains two collaborative components:

- **VGAP (Visual Grounder with Adaptive Prototype):** extracts adaptive prototypes from shallow, middle, and deep visual tokens and injects them into image-conditional prompts through cross-attention.
- **SCD (Synergistic Causal Disentanglement):** separates causal and non-causal prompt pathways and combines classification, non-causal entropy maximization, and orthogonality objectives.

On 11 datasets across 9 biomedical imaging modalities, BiomedCCPL achieves an average base-to-novel harmonic mean of **79.98%**.

## Results

| Setting | Metric | Result |
|---|---:|---:|
| 1-shot | Accuracy | 62.17 |
| 2-shot | Accuracy | 64.86 |
| 4-shot | Accuracy | 71.49 |
| 8-shot | Accuracy | 77.22 |
| 16-shot | Accuracy | 82.25 |
| Base-to-novel | Base / Novel / HM | 80.78 / 79.20 / 79.98 |

Results are averaged over the datasets and three random seeds as described in the paper.

## Repository structure

```text
BiomedCCPL/
├── configs/                  # Dataset- and shot-specific hyperparameters
├── datasets/                 # Unified loader for 11 biomedical datasets
├── scripts/                  # Reproduction scripts
├── tools/                    # Result aggregation and ADR-0001 diagnostics
├── trainers/biomedccpl.py    # VGAP, SCD, and the trainer
├── train.py                  # Training/evaluation entry point
└── requirements.txt
```

## Installation

The experiments in the paper use Python 3.10, a ViT-B/16 BiomedCLIP backbone, and a single NVIDIA GeForce RTX 4090 GPU.

```bash
conda create -n biomedccpl python=3.10 -y
conda activate biomedccpl

# Select the PyTorch/CUDA build suitable for your machine.
# Example for CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Install Dassl separately following the standard CoOp/Dassl setup (skip this step if it is already available in your environment):

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

The first run downloads `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` from Hugging Face. Set `HF_ENDPOINT` yourself if a mirror is required.

### Local development and tests (macOS, CPU)

The tests in `tests/` run on CPU and need neither Dassl nor the datasets; they need the BiomedCLIP weights (downloaded on first use). Tested on an Apple M2 with Homebrew Python 3.12 (`numpy<2` has no wheels for Python 3.13):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests
```

To run `train.py`, also install Dassl into the same environment. Its `setup.py` imports numpy, so build it without isolation, and skip its pinned linters (`--no-deps`), installing only what it imports at runtime:

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
.venv/bin/pip install --no-deps --no-build-isolation -e Dassl.pytorch
.venv/bin/pip install tabulate gdown "pandas<3" wilds==1.2.2
```

`Dassl.pytorch/` is git-ignored if you clone it inside the repo.

`tests/test_vgap_modes.py` checks that `TRAINER.BIOMEDCCPL.VGAP.MODE text` (the default) reproduces the original model exactly, by comparing against `trainers/biomedccpl.py` as it was at commit `3257176` (see `docs/adr/0001-image-side-prototype-readout.md`).

## Data preparation

Please follow the official BiomedCoOp [DATASETS.md](https://github.com/HealthX-Lab/BiomedCoOp/blob/main/assets/DATASETS.md) to download, extract, and organize the datasets. BiomedCCPL uses the same 11 biomedical classification datasets and data splits. After completing those instructions, the dataset root passed to `--root` should contain:

```text
data/
├── BTMRI/BTMRI/<class_name>/*.jpg
├── BUSI/BUSI/<class_name>/*
├── CHMNIST/CHMNIST/<class_name>/*
├── COVID_19/COVID_19/<class_name>/*
├── CTKidney/CTKidney/<class_name>/*
├── DermaMNIST/DermaMNIST/<class_name>/*
├── KneeXray/KneeXray/<class_name>/*
├── Kvasir/Kvasir/<class_name>/*
├── LungColon/LungColon/<class_name>/*
├── OCTMNIST/OCTMNIST/<class_name>/*
└── RETINA/RETINA/<class_name>/*
```


## Training and evaluation

Run all commands from the repository root.

### Few-shot evaluation

The script trains and evaluates seeds 1, 2, and 3, then reports their mean and standard deviation:

```bash
bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0
```

Arguments are `DATA_ROOT`, `DATASET`, `SHOTS`, and optional `GPU_ID`. Valid shot counts are 1, 2, 4, 8, and 16. Results go to `output/few_shot/text/<DATASET>/<SHOTS>shot/seed<N>`; see [ADR-0001 variants](#adr-0001-variants) for the other model variants.

### Base-to-novel generalization

```bash
bash scripts/run_base_to_novel.sh /path/to/data BTMRI 0
```

This trains on the base classes with 16 shots per class and evaluates the same checkpoint on novel classes for three seeds. Following the paper, BUSI is excluded because it has only three classes.

### ADR-0001 variants

[ADR-0001](docs/adr/0001-image-side-prototype-readout.md) compares the original model with variants selected by `TRAINER.BIOMEDCCPL.VGAP.MODE`. Both scripts take the mode from `VGAP_MODE` (default `text`), write to `output/<protocol>/<RUN_TAG>/...` (`RUN_TAG` defaults to the mode), and pass any arguments after `GPU_ID` to `train.py` as config overrides:

```bash
bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0                     # R0: original BiomedCCPL
VGAP_MODE=none bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0      # R1: no VGAP
VGAP_MODE=image bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0     # R2: image-side readout
VGAP_MODE=image RUN_TAG=r3 bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0 \
  TRAINER.BIOMEDCCPL.VGAP.IMAGE_LAYERS "[3, 7, 11]"                       # R3
VGAP_MODE=image RUN_TAG=r2_lr10 bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0 \
  TRAINER.BIOMEDCCPL.VGAP.READOUT_LR_MULT 10.0                            # R2, readout at 10x LR
VGAP_MODE=text_agnostic bash scripts/run_few_shot.sh /path/to/data BTMRI 16 0  # R4: shared VGAP query
```

Set a distinct `RUN_TAG` whenever two runs share a mode, or the second overwrites the first. `run_base_to_novel.sh` works the same way and applies the mode and overrides to both the base training and the novel evaluation.

In `MODE image` each training run also writes `readout_attention.pt` to its output folder: the prototype maps (14×14 patch grid) and readout attention for 20 fixed test images, with labels, predictions and the final gate β. Outside `MODE text`/`text_agnostic`, evaluation encodes the class prompts once and reuses them for every batch.

Diagnostics and cost tools (inference or synthetic inputs only):

```bash
# Phase 0 (ADR §5) on a trained R0 checkpoint -> results/phase0/BTMRI_k16_seed1.json
python tools/phase0_diagnostics.py --root /path/to/data --seed 1 \
  --config-file configs/trainers/BiomedCCPL/few_shot/16/BTMRI.yaml \
  --model-dir output/few_shot/text/BTMRI/16shot/seed1 --load-epoch 50 \
  DATASET.NAME BTMRI DATASET.NUM_SHOTS 16

# Re-create readout_attention.pt from an R2/R3 checkpoint
python tools/save_readout_attention.py --root /path/to/data --seed 1 \
  --output-dir output/attention/BTMRI/seed1 \
  --config-file configs/trainers/BiomedCCPL/few_shot/16/BTMRI.yaml \
  --model-dir output/few_shot/image/BTMRI/16shot/seed1 --load-epoch 50 \
  DATASET.NAME BTMRI DATASET.NUM_SHOTS 16 TRAINER.BIOMEDCCPL.VGAP.MODE image

# Cost benchmark (ADR §7.3) on synthetic BTMRI-sized batches -> results/cost/BTMRI_cuda.json
python tools/benchmark_cost.py --device cuda --modes text image --batch-sizes 1 8 32
```

### Single run

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --root /path/to/data \
  --seed 1 \
  --output-dir output/btmri/seed1 \
  --config-file configs/trainers/BiomedCCPL/few_shot/16/BTMRI.yaml \
  DATASET.NAME BTMRI \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES all
```

Configuration values can be overridden at the end of the command, for example:

```bash
TRAINER.BIOMEDCCPL.ALPHA 0.3 \
TRAINER.BIOMEDCCPL.PROTONUM 14 \
TRAINER.BIOMEDCCPL.CROSSLAYERS "[3, 7, 11]"
```

### Evaluate a checkpoint

```bash
python train.py \
  --root /path/to/data \
  --output-dir output/eval \
  --config-file configs/trainers/BiomedCCPL/base_to_novel/BTMRI.yaml \
  --model-dir output/base_to_novel/text/BTMRI/base/seed1 \
  --load-epoch 50 \
  --eval-only \
  DATASET.NAME BTMRI \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES new
```

Evaluate with the same `TRAINER.BIOMEDCCPL.VGAP.MODE` the checkpoint was trained with; loading prints a warning if the checkpoint's keys do not match the model.

## Citation

```bibtex
@InProceedings{Cui_2026_CVPR,
  author    = {Cui, Xueliang and Zhang, Juncai and Hou, Jiacheng and Lu, Dan and Zhang, Hao and Wang, Ruxin},
  title     = {BiomedCCPL: Causal Conditional Prompt Learning for Biomedical Vision-Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {40812--40821}
}
```

## Acknowledgements

This codebase is built on [CoOp](https://github.com/KaiyangZhou/CoOp) and [BiomedCoOp](https://github.com/HealthX-Lab/BiomedCoOp). We sincerely thank the authors for publicly releasing their code. If you find our model or code useful, we encourage you to acknowledge and cite these foundational works as well.
