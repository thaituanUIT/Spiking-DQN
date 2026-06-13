# SpikingDQN: Applying SNN to RL in Active Object Localization

This repository explores the intersection of Reinforcement Learning (RL) and Spiking Neural Networks (SNNs) to perform Active Object Localization. 

By formulating object localization as a Markov Decision Process (MDP), our RL agent learns to sequentially adjust a bounding box—moving, scaling, and altering its aspect ratio—until it accurately encapsulates the target object. The agent's policy and value networks are powered by Spiking Neural Networks, leveraging their innate temporal dynamics and efficiency.

## Project Status

**Active & Stable**: The project is under active development. The `v2` codebase has been fully stabilized, effectively addressing catastrophic forgetting during RL training with deep feature extractors (e.g., VGG16, ResNet18) using Global Average Pooling. The agent's reward structure has been optimized to prevent local optima, and the SNN conversion (`ats`) and simulation logic are functionally complete and ready for full-scale experiments.

## Repository Structure

The repository is now consolidated into one active codebase:

*   `cli.py`: Unified command line entrypoint for the baseline and SNN pipelines.
*   `baseline/`: Standard, non-spiking DQN baseline for comparison.
*   `v2/`: Active SNN + RL implementation and the main research path.

Deprecated branches `v1/` and `v3/` are no longer part of the active workflow.

## Dataset

The CLI now supports two datasets through a single flag:

- `--dataset voc`
- `--dataset tiny-imagenet`

All dataset utilities use the repo-local `dataset/` folder automatically. If the expected dataset files are already present there, they are reused. If Tiny ImageNet is missing, it is downloaded automatically into `dataset/tiny-imagenet-200/`.

Expected layout:
```text
SpikingDQN/
├── cli.py
├── baseline/
├── v2/
└── dataset/
    ├── VOC2012/
    │   ├── Annotations/
    │   └── JPEGImages/
    └── tiny-imagenet-200/
        ├── train/
        └── val/
```

Notes:

- VOC training still uses `torchvision` and expects the standard VOC folder layout under `dataset/`.
- VOC evaluation uses TFDS VOC2007 test data.
- Tiny ImageNet training and evaluation use Tiny ImageNet bounding-box annotations from the extracted archive.
- A legacy Tiny ImageNet checkout under `IMagenet/tiny-imagenet-200/` is still detected as a fallback.

## Unified CLI

Use a single command surface from the repository root:

```bash
python3 cli.py <baseline|v2> <train|test|render> [options]
```

Examples:

```bash
python3 cli.py baseline train --dataset voc --target aeroplane --extractor vgg16 --epochs 10
python3 cli.py v2 train --method surrogate --dataset tiny-imagenet --target mixing --epochs 20
python3 cli.py v2 test --method ats --dataset voc --target mixing --weights weights/ats_mixing.pth
```

## Quick Start (v2 Architecture)

The `v2` framework is designed for ease of use. It supports 3 primary SNN Methods:

1.  **Surrogate** (`--method surrogate`): Spiking Convolutional layers trained End-to-End via BPTT with surrogate gradients (`SuperSpike`).
2.  **ATS** (`--method ats`): ANN-to-SNN. Trains as a regular CNN and simulates discrete Integration-and-Fire neurons during evaluation.
3.  **STDP** (`--method stdp`): Unsupervised bio-plausible Feature Extraction using Difference of Gaussians (DoG) and Winner-Take-All lateral inhibition, followed by a spiking RL head.

*(Note: `surrogate` and `ats` optionally support a frozen VGG16 backbone via `--extractor vgg16` for abstracted feature extraction).*

### Training

Use `cli.py v2 train` to train an agent. You can specify a target class or use `mixing` to train on all objects.
The training pipeline also supports early stopping (`--early-stop`) and saving the best model (`--save best`).

```bash
# Train using Surrogate Gradients to localize aeroplanes (with early stopping and best model saving)
python3 cli.py v2 train --method surrogate --dataset voc --target aeroplane --epochs 20 --early-stop 5 --save best

# Train the ATS model using a VGG16 backbone
python3 cli.py v2 train --method ats --dataset voc --target mixing --extractor vgg16 --epochs 50

# Train on Tiny ImageNet
python3 cli.py v2 train --method surrogate --dataset tiny-imagenet --target mixing --epochs 20
```

### Evaluation & Testing

Use `cli.py v2 test` to evaluate your saved models. The script calculates Localization Accuracy at multiple Intersection-over-Union (IoU) thresholds.

```bash
# Evaluate with visual Matplotlib playback of the bounding box search path
python3 cli.py v2 render --method surrogate --dataset voc --target aeroplane

# Evaluate quietly and export granular metrics to a CSV file
python3 cli.py v2 test --method surrogate --dataset tiny-imagenet --target mixing --logging-dir logs
```

---
*Developed as part of research into Advanced Agentic Coding and Spiking Reinforcement Learning.*
