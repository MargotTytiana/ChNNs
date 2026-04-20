# ChaoNet: Chaotic Hierarchical Network for Speaker Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Research-orange" />
</p>

> **ChaoNet** is a novel speaker recognition framework that replaces conventional spectral features with nonlinear dynamical features derived from chaos theory. Instead of asking *what the voice sounds like*, ChaoNet asks *how the voice is produced* — modeling the speaker's vocal tract as a chaotic dynamical system and learning to recognize speakers from the geometry of their strange attractors.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Key Components](#key-components)
  - [Phase Space Reconstruction](#1-phase-space-reconstruction)
  - [Chaotic Feature Extraction](#2-chaotic-feature-extraction-mlsa--rqa)
  - [Chaotic Embedding Layer](#3-chaotic-embedding-layer)
  - [Strange Attractor Pooling](#4-strange-attractor-pooling)
  - [Speaker Embedding & Classification](#5-speaker-embedding--classification)
- [Supported Chaotic Systems](#supported-chaotic-systems)
- [Results](#results)
  - [Clean Conditions](#clean-conditions)
  - [Noise Robustness](#noise-robustness)
  - [Ablation Studies](#ablation-studies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training ChaoNet](#training-chaonet)
  - [Training Baselines](#training-baselines)
  - [Configuration](#configuration)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Motivation

Conventional speaker recognition systems rely on Mel-spectrogram or MFCC features, which model speech production as a **linear, stationary process**. This assumption breaks down in two important ways:

1. **Physiological reality**: Vocal fold vibration is governed by nonlinear differential equations. The source–filter coupling produces deterministic chaos — small individual differences in subglottal pressure, muscle tension, and tissue properties are amplified into unique acoustic signatures. MFCC discards precisely this information.

2. **Noise sensitivity**: Spectral features are computed frame-by-frame and directly reflect instantaneous spectral content. When noise corrupts a frame, there is no recovery mechanism. Chaotic invariants (e.g., Lyapunov exponents, correlation dimension) are computed over entire trajectories, so random perturbations average out.

Empirical evidence supports this: normal voiced speech consistently shows a **positive maximum Lyapunov exponent** (λ_max ≈ 0.3–0.6 bit/ms), confirming low-dimensional chaotic dynamics. Phase space reconstruction of different speakers reveals distinct strange attractor geometries — measurably different topological structures that carry speaker identity even when spectral content is noise-corrupted.

ChaoNet builds on this foundation by integrating chaotic dynamics directly into a trainable neural network pipeline.

---

## Architecture Overview

```
Raw Audio Waveform
        │
        ▼
┌─────────────────────────┐
│  Phase Space            │  Takens' embedding theorem
│  Reconstruction         │  τ = 12 samples, d_e = 10
└──────────┬──────────────┘
           │  [batch, 200, 10]
           ▼
┌──────────────────────────────────────────┐
│  Chaotic Feature Extraction              │
│  ┌─────────────────┐ ┌────────────────┐  │
│  │  MLSA (105-dim) │ │  RQA (125-dim) │  │  Total: 230-dim
│  │  Multi-scale    │ │  Recurrence    │  │
│  │  Lyapunov       │ │  Quantification│  │
│  └─────────────────┘ └────────────────┘  │
└──────────────────┬───────────────────────┘
                   │  [batch, 230]
                   ▼
┌─────────────────────────┐
│  Chaotic Embedding      │  Controlled Lorenz/Rossler/
│  Layer                  │  Mackey-Glass/Chua neurons
│  + Bifurcation Control  │  RK4 integration, T=0.5s
└──────────┬──────────────┘
           │  [batch, 50, 3]  (trajectory on attractor)
           ▼
┌─────────────────────────┐
│  Strange Attractor      │  117 topological invariants:
│  Pooling                │  D₂, D_L, K, statistical
│                         │  moments, FFT, autocorr, ...
└──────────┬──────────────┘
           │  [batch, 117]
           ▼
┌─────────────────────────┐
│  Speaker Embedding      │  [512 → 256 → 128] MLP
│  Network                │  + L2 normalization
└──────────┬──────────────┘
           │  [batch, 256]
           ▼
┌─────────────────────────┐
│  Classifier             │  ArcFace (s=30, m=0.35)
│  (ArcFace / Linear)     │  or Linear + Cross-Entropy
└─────────────────────────┘
```

The entire pipeline is end-to-end differentiable. Gradients propagate through the classifier, pooling layer, and trajectory generation back into the feature extraction networks, allowing the model to learn which chaotic dynamics best discriminate speakers.

---

## Key Components

### 1. Phase Space Reconstruction

**File**: `core/phase_space_reconstruction.py`

Implements Takens' embedding theorem to reconstruct a high-dimensional phase space from the raw 1D audio signal:

```
x(t) = [s(t), s(t+τ), s(t+2τ), ..., s(t+(d_e−1)τ)]
```

Parameter selection uses:
- **Time delay τ**: First minimum of the mutual information function (autocorrelation fallback)
- **Embedding dimension d_e**: False Nearest Neighbors (FNN) algorithm

Ablation experiments on 100 training utterances determined **τ = 12** samples and **d_e = 10** as the optimal fixed parameters for LibriSpeech 16 kHz audio.

Supported estimation methods:
| Method | Class | Notes |
|--------|-------|-------|
| Autocorrelation | `AutocorrelationDelayEstimator` | Fast, linear |
| Mutual Information | `MutualInformationDelayEstimator` | Captures nonlinear deps |
| False Nearest Neighbors | `FalseNearestNeighborsDimensionEstimator` | Standard for d_e |
| Cao's method | `CaoDimensionEstimator` | Alternative for d_e |

---

### 2. Chaotic Feature Extraction (MLSA + RQA)

#### Multi-scale Lyapunov Spectrum Analysis (MLSA)

**File**: `core/mlsa_extractor.py`

Analyzes chaotic dynamics at 5 temporal scales (k ∈ {1, 2, 4, 8, 16}) using wavelet decomposition. At each scale, extracts:

| Feature | Description |
|---------|-------------|
| λ_max | Largest Lyapunov exponent (Rosenstein algorithm) |
| D₂ | Correlation dimension (Grassberger–Procaccia) |
| H | Hurst exponent (long-range correlations) |
| H_λ | Lyapunov spectral entropy |
| Sample entropy | Complexity measure |
| Path statistics | Trajectory path length and complexity |

Each feature is aggregated with 6 statistics (mean, std, min, max, range, scale-weighted average), producing **105 total MLSA dimensions**.

#### Recurrence Quantification Analysis (RQA)

**File**: `core/rqa_extractor.py`

Constructs recurrence matrices using an adaptive threshold (10th percentile of pairwise distances) and extracts:

| Measure | Symbol | Description |
|---------|--------|-------------|
| Recurrence Rate | RR | Frequency of state recurrence |
| Determinism | DET | Fraction of recurrence in diagonal lines |
| Laminarity | LAM | Fraction of recurrence in vertical lines |
| Mean diagonal length | L_mean | Duration of deterministic behavior |
| + 13 additional metrics | — | Trapping time, entropy, RPDE, etc. |

Multi-scale analysis over 4 scales with 7-statistic aggregation produces **125 total RQA dimensions**.

**Combined chaotic feature vector: 230 dimensions.**

---

### 3. Chaotic Embedding Layer

**File**: `core/chaotic_embedding.py`

Maps the 230-dim feature vector to initial conditions and coupling parameters of a chaotic dynamical system, then evolves the system forward using 4th-order Runge–Kutta integration:

```python
# Controlled Lorenz system
dx/dt = σ(y − x) + c_x
dy/dt = x(ρ − z) − y + c_y
dz/dt = xy − βz + c_z
```

Three learnable networks handle the mapping:

| Network | Architecture | Output | Purpose |
|---------|-------------|--------|---------|
| `initial_state_mapper` | 230→16→3 (Tanh) | x₀ ∈ [−2, 2]³ | Starting point on attractor |
| `coupling_mapper` | 230→8→3 (Tanh) | c ∈ R³ | Persistent speaker-specific forcing |
| `param_adapter` | 230→8→3 (Sigmoid) | (s_σ, s_ρ, s_β) | Modulates Lorenz parameters |

**Integration**: T = 0.5 s, h = 0.01 → trajectory shape `[batch, 50, 3]`

#### Bifurcation Control

An optional lightweight network (2 layers, ~3,700 parameters) predicts a regime signal r ∈ [0,1] that **multiplicatively modulates** ρ by ±20%:

```
ρ_modulate = ρ_base · (0.8 + 0.4 · r)
```

This allows the model to explore different dynamical regimes of the Lorenz attractor while preserving the learned parameter adapter mapping. Replacing rather than modulating ρ causes catastrophic accuracy collapse (94.07% → 5.93%).

---

### 4. Strange Attractor Pooling

**File**: `core/attractor_pooling.py`

Converts the variable-structure trajectory `[batch, 50, 3]` into a fixed 117-dimensional feature vector by extracting topological invariants robust to time shifts and small perturbations:

| Category | Description | Dim |
|----------|-------------|-----|
| Topological invariants | D₂, D_L, K, gyration radius, box dimension | 5 |
| Statistical moments | Mean, std, skewness, kurtosis per axis | 12 |
| Extrema | Min and max per axis | 6 |
| Velocity statistics | Mean, std, max of ẋ per axis | 9 |
| Acceleration statistics | Mean, std, max of ẍ per axis | 9 |
| Cross-correlations | ρ_xy, ρ_xz, ρ_yz | 3 |
| Path length | Total trajectory arc length | 1 |
| Percentiles | 10th–90th per axis | 15 |
| Temporal samples | 10 fixed-interval samples × 3 axes | 30 |
| Autocorrelation | 3 lags × 3 axes | 9 |
| Frequency features | FFT power in 6 bins × 3 axes | 18 |
| **Total** | | **117** |

Three pooling modes are available: `basic` (3-dim), `comprehensive` (117-dim), `learnable` (3-dim with learned weights).

---

### 5. Speaker Embedding & Classification

**File**: `models/chaotic_network.py`

The pooled 117-dim features pass through an enhanced speaker embedding network:

```
117 → [512 → BN → ReLU → Dropout(0.3)]
    → [256 → BN → ReLU → Dropout(0.24)]
    → [128 → BN → ReLU → Dropout(0.19)]
    → 256 (L2-normalized embedding)
```

Classification uses **ArcFace** angular margin loss (scale s = 30, margin m = 0.35) during training, and cosine similarity at inference:

```
L_AM = −(1/N) Σ log [exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_{j≠yi} exp(s·cos(θ_j)))]
```

---

## Supported Chaotic Systems

| System | Type | Key Property | Parameters |
|--------|------|--------------|------------|
| **Lorenz** | 3D ODE | Double-scroll attractor, broadband | σ=10, ρ=28, β=8/3 |
| **Rössler** | 3D ODE | Single-scroll, computationally efficient | a=0.2, b=0.2, c=5.7 |
| **Mackey–Glass** | DDE | Delay differential, temporal memory | β=0.2, γ=0.1, τ=17, n=10 |
| **Chua** | 3D ODE | Multi-scroll, models abrupt transitions | α=10, β=14.87 |

Each system has system-specific parameter optimization ranges validated in `core/chaos_utils.py`.

---

## Results

### Clean Conditions

#### 26-Speaker Evaluation (dev-clean-2, 679 samples)

| Model | Parameters | Test Accuracy | Test Loss |
|-------|-----------|--------------|-----------|
| Mel-MLP | 69,284 | **100.00%** | 0.0527 |
| MFCC-MLP | 15,876 | 94.07% | 0.4602 |
| Mel-CNN | 185,380 | 99.15% | 0.0414 |
| MFCC-CNN | 172,516 | 98.31% | 0.1113 |
| **ChaoNet** | **521,000+** | 94.07% | — |

#### 251-Speaker Evaluation (train-clean-100, 7,526 samples)

| Model | Parameters | Test Accuracy | Test Loss |
|-------|-----------|--------------|-----------|
| Mel-MLP | 79,099 | **96.80%** | 0.1332 |
| MFCC-MLP | 30,501 | 76.40% | 0.9960 |
| Mel-CNN | — | 95.20% | 0.1620 |
| MFCC-CNN | — | 90.40% | 0.3202 |
| **ChaoNet** | **578,111** | 92.75% | 0.2694 |

ChaoNet is competitive but not the top performer under clean conditions — consistent with the design goal of prioritizing robustness over clean-condition accuracy.

---

### Noise Robustness

Evaluation on 26-speaker test set with four noise types at five SNR levels (20–0 dB).

#### Accuracy (%) under Gaussian Noise

| SNR | ChaoNet | Mel-MLP | MFCC-MLP |
|-----|---------|---------|---------|
| Clean | 94.92 | 94.92 | 99.15 |
| 20 dB | **77.12** | 22.88 | 27.12 |
| 15 dB | **59.32** | 13.56 | 11.02 |
| 10 dB | **24.58** | 10.17 | 11.02 |
| 5 dB | 10.17 | 5.93 | 7.63 |
| 0 dB | 4.24 | 4.24 | 4.24 |

#### Performance Retention Rate at 20 dB SNR

| Noise Type | ChaoNet | Mel-MLP | MFCC-MLP |
|-----------|---------|---------|---------|
| Gaussian | **81.2%** | 24.1% | 27.4% |
| Babble | **79.5%** | 51.8% | 62.4% |
| Café | **81.2%** | 24.1% | 26.5% |
| Street | **88.4%** | 39.3% | 58.1% |
| **Average** | **82.6%** | 34.8% | 43.6% |

ChaoNet retains **82.6%** of clean-condition accuracy at 20 dB SNR, compared to 34.8% (Mel-MLP) and 43.6% (MFCC-MLP) — approximately **1.9–2.4× more robust**.

The advantage is largest under Gaussian and Café noise (broadband corruption), and somewhat smaller under Babble noise (spectrally similar interference).

---

### Ablation Studies

#### Component Contribution (26-speaker)

| Configuration | Val Acc | Test Acc |
|--------------|---------|---------|
| Full Model | 97.75% | — |
| w/o Attractor Pooling | 95.51% | 88.14% |
| w/o Chaotic Features | 96.63% | 90.68% |
| w/o Chaotic Embedding | 96.63% | 86.44% |
| Minimal (no chaotic components) | 94.38% | 93.22% |

#### Bifurcation Control Strategy

| Strategy | Mechanism | Test Accuracy |
|----------|-----------|--------------|
| No bifurcation control | — | 94.07% |
| Replacement | ρ ← 24 + 8r (overrides adapter) | **5.93%** |
| **Modulation** | ρ ← ρ_base · (0.8 + 0.4r) | **94.92%** |

#### Early Stopping Sensitivity

| Patience | Stop Epoch | Best Accuracy |
|----------|-----------|--------------|
| 5 | 17 | 21.35% |
| 10 | 33 | 41.57% |
| ≥15 | 99 | **96.63%** |

The cosine annealing schedule requires patience ≥ 15 to capture continued improvement in late training epochs.

---

## Project Structure

```
ChNNs/
├── core/
│   ├── phase_space_reconstruction.py  # Takens' embedding, FNN, mutual info
│   ├── mlsa_extractor.py              # Multi-scale Lyapunov Spectrum Analysis
│   ├── rqa_extractor.py               # Recurrence Quantification Analysis
│   ├── chaotic_embedding.py           # Lorenz/Rossler/MG/Chua neurons + RK4
│   ├── attractor_pooling.py           # 117-dim topological invariant extraction
│   └── chaos_utils.py                 # Lyapunov calculation, system validation
│
├── models/
│   ├── chaotic_network.py             # Full ChaoNet architecture
│   ├── hybrid_models.py               # TraditionalMLPBaseline, CNN baselines
│   ├── mlp_classifier.py              # MLP classifier module
│   └── model_factory.py              # Factory functions
│
├── features/
│   ├── traditional_features.py        # MelSpectrogramExtractor, MFCCExtractor
│   ├── chaotic_features.py            # ChaoticFeatureExtractor wrapper
│   └── differentiable_chaos_features.py  # Fully differentiable MLSA/RQA approx
│
├── data/
│   ├── dataset_loader.py              # LibriSpeech loader, collate functions
│   ├── audio_preprocessor.py          # RMS normalization, silence trimming
│   └── data_utils.py                  # DataValidator, DatasetSplitter
│
├── experiments/
│   ├── base_experiment.py             # Abstract training loop
│   ├── chaotic_experiment.py          # ChaoNet training + sync loss + adversarial
│   ├── baseline_experiment.py         # Baseline training pipeline
│   └── configs/
│       ├── base_config.yaml
│       ├── chaotic_config.yaml
│       └── comparison_config.yaml
│
├── scripts/
│   ├── train_chaotic.py               # Main training script (CLI)
│   ├── train_baselines.py             # Baseline training script
│   ├── diagnose_chaotic_training.py   # Gradient flow diagnostic
│   └── sbatch_files/                  # SLURM job scripts for CSC Mahti
│
├── evaluation/
│   └── metrics.py                     # MetricsCalculator, StatisticalAnalyzer
│
├── utils/
│   ├── checkpoint.py                  # CheckpointManager
│   ├── logger.py                      # Experiment logging setup
│   ├── reproducibility.py             # set_seed, get_system_info
│   └── numerical_stability.py         # NumericalConfig, OutlierDetector
│
└── sync_loss_extensions.py            # ScheduledSyncLoss, HierarchicalSyncLoss,
                                       # SyncLossWithGradientSurgery
```

---

## Installation

### Requirements

- Python 3.12+
- PyTorch 2.7.1+ with CUDA 12.6 (CPU also supported)
- NVIDIA GPU recommended (A100 used in experiments)

### Setup

```bash
git clone https://github.com/MargotTytiana/ChNNs.git
cd ChNNs
git checkout origin

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy scikit-learn librosa soundfile matplotlib pandas pywt tqdm pyyaml
```

### Optional dependencies

```bash
# For Jupyter notebooks
pip install jupyterlab

# For TensorBoard logging
pip install tensorboard

# For t-SNE visualization in embedding analysis
pip install scikit-learn  # already included above
```

---

## Usage

### Training ChaoNet

```bash
# Train with Lorenz system (default configuration)
python scripts/train_chaotic.py \
    --system lorenz \
    --model_type full_chaotic \
    --data_dir /path/to/LibriSpeech/train-clean-100 \
    --num_speakers 251 \
    --epochs 100 \
    --output_dir ./outputs/chaotic

# Train all four chaotic systems
python scripts/train_chaotic.py \
    --all \
    --data_dir /path/to/LibriSpeech \
    --epochs 100

# Train with custom config file
python scripts/train_chaotic.py \
    --config experiments/configs/chaotic_config.yaml \
    --output_dir ./outputs/chaotic
```

### Training Baselines

```bash
# Train all four baselines (Mel-MLP, MFCC-MLP, Mel-CNN, MFCC-CNN)
python scripts/train_baselines.py \
    --data_dir /path/to/LibriSpeech/train-clean-100 \
    --num_speakers 251 \
    --output_dir ./outputs/baselines

# Train a specific baseline
python scripts/train_baselines.py \
    --baseline_type mel_mlp \
    --data_dir /path/to/LibriSpeech
```

### Configuration

Key parameters in `experiments/configs/chaotic_config.yaml`:

```yaml
# Chaotic system selection
model:
  chaotic_network:
    embedding:
      system_type: "lorenz"      # lorenz | rossler | mackey_glass | chua
      evolution_time: 1.0        # seconds of chaotic evolution
      coupling_strength: 2.0     # strength of feature-to-system coupling

# Phase space reconstruction
features:
  chaotic:
    mlsa:
      scales: [1, 2, 4, 8, 16]   # temporal scales for MLSA
      max_lyapunov_steps: 2000

# Training
training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 200
  optimizer:
    type: "adamw"
    adamw:
      weight_decay: 1e-4
  scheduler:
    type: "plateau"
    plateau:
      patience: 50
      factor: 0.7
```

### Resuming from Checkpoint

```bash
python scripts/train_chaotic.py \
    --system lorenz \
    --resume ./outputs/chaotic/models/lorenz_full_chaotic_run_0.pth \
    --epochs 200
```

### Python API

```python
from models.chaotic_network import ChaoticSpeakerRecognitionNetwork
import torch

# Initialize model
model = ChaoticSpeakerRecognitionNetwork(
    sample_rate=16000,
    embedding_dim=10,          # phase space embedding dimension
    mlsa_scales=5,             # number of temporal scales
    chaotic_system='lorenz',   # chaotic system type
    evolution_time=0.5,        # trajectory length in seconds
    pooling_type='comprehensive',
    speaker_embedding_dim=256,
    num_speakers=251,
    classifier_type='linear',
    use_bifurcation_control=True,
    device='cuda'
)

# Forward pass
audio = torch.randn(4, 48000)   # batch of 4 utterances, 3s at 16kHz
logits = model(audio)           # [4, 251]

# Extract embeddings for verification
embeddings = model.extract_embeddings(audio)  # [4, 256]

# Predict with confidence
predicted_speakers, confidence = model.predict(audio)
```

### Running on HPC (SLURM / CSC Mahti)

```bash
# Submit a GPU job
sbatch scripts/sbatch_files/gpu_main_lorenz.sh

# Resume a previous run
sbatch scripts/sbatch_files/gpu_main_lorenz_resume.sh
```

Example SLURM script (`gpu_main_lorenz.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=chaonet_lorenz
#SBATCH --account=project_2003370
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G

module load pytorch
apptainer run --nv pytorch-2.7.sif \
    python scripts/train_chaotic.py \
    --system lorenz \
    --data_dir /scratch/project_2003370/yueyao/dataset/LibriSpeech \
    --output_dir /scratch/project_2003370/yueyao/outputs/chaotic
```

---

## Dataset

Experiments use the **LibriSpeech** corpus:

| Split | Speakers | Hours | Used for |
|-------|---------|-------|---------|
| `dev-clean-2` | 26 | ~2 | Small-scale evaluation |
| `train-clean-100` | 251 | 100 | Large-scale evaluation |

**Preprocessing pipeline**:
1. Resample to 16 kHz (if necessary)
2. RMS normalization to target level 0.1
3. Silence trimming (top_db = 25 dB)
4. Truncate/pad to 3.0 seconds
5. Train/val/test split: 70% / 15% / 15% (file-level split)

**Data loading**:

```python
from data.dataset_loader import create_speaker_dataloaders

train_loader, val_loader, test_loader = create_speaker_dataloaders(
    data_dir='/path/to/LibriSpeech/train-clean-100',
    batch_size=32,
    sample_rate=16000,
    max_length=3.0,
    num_workers=4,
    target_num_speakers=251,
    seed=42
)
```

---

## Reproducibility

All experiments use fixed random seed 42 across Python, NumPy, and PyTorch:

```python
from utils.reproducibility import set_seed
set_seed(42)
```

System information is logged automatically at the start of each experiment:

```python
from utils.reproducibility import get_system_info
print(get_system_info())
# Platform, GPU, CUDA version, package versions, git commit hash
```

Experiments were run on **CSC Mahti** (NVIDIA A100-SXM4-40GB, CUDA 12.6, PyTorch 2.7.1).

---

## Citation

If you use ChaoNet in your research, please cite:

```bibtex
@mastersthesis{tian2026chaonet,
  author    = {Tian, Yueyao},
  title     = {{ChaoNet}: Chaotic Hierarchical Network for Speaker Recognition},
  school    = {Tampere University},
  year      = {2026},
  month     = {March},
}
```

---

## Acknowledgements

This work was carried out at Tampere University under the supervision of Prof. Annamaria Mesaros. Computational resources were provided by CSC – IT Center for Science, Finland (project `project_2003370`).

The theoretical foundations draw on:
- Takens (1981) — Phase space reconstruction theorem
- Pecora & Carroll (1990) — Chaotic synchronization
- Grassberger & Procaccia (1983) — Correlation dimension
- Rosenstein et al. (1993) — Practical Lyapunov exponent estimation
- Marwan et al. (2007) — Recurrence quantification analysis

---

## License

This project is released under the MIT License. See `LICENSE` for details.
