# Generative Modeling on CIFAR-10

## Project Overview

This project implements and compares three different generative model architectures on the CIFAR-10 dataset:
- **Variational Autoencoders (VAE)**
- **Energy-Based Models (EBM)**
- **Noise Conditional Score Network (NCSN)**

CIFAR-10 is a standard benchmark in generative modeling research, widely used in papers on VAE, EBM, Score-Based, and Diffusion models. The dataset is diverse enough to be challenging, yet manageable for model training with 32×32 RGB images from real-world scenes.

**Important Note:** While CIFAR-10 contains class labels, they are **not used** in this project. The focus is on unsupervised learning of the data distribution p(x).

### Project Goals
- Implement all models from scratch (Loss functions, Sampling procedures, Training loops)
- Conduct 2 experiments: one with NCSN (required) and one with VAE/EBM
- Analyze scientific literature for each architecture
- Compare different generative architectures on the same dataset

## 1. Data Exploration

notebooks/data_exploration.ipynb

### Dataset Overview
CIFAR-10 consists of 60,000 32×32 color images in 10 classes:
- **Training set**: 50,000 images (5,000 per class)
- **Test set**: 10,000 images (1,000 per class)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Format**: 32×32 RGB images
- **Pixel range**: [0, 255]

### Key Findings from Exploration
The dataset exhibits the following characteristics:
- **Balanced distribution**: Equal number of samples per class
- **Diverse content**: Real-world images with varying lighting, angles, and backgrounds
- **Image quality**: Low resolution (32×32) presents challenges for fine-grained detail generation

## 2. Project Structure

```
├── src/                    # Source code for all models
│   ├── vae/               # Variational Autoencoder implementation
│   ├── ebm/               # Energy-Based Model implementation
│   ├── ncsn/              # Noise Conditional Score Network implementation
│   └── utils/             # Data loading, metrics, visualization
├── notebooks/             # Jupyter notebooks for experiments
├── data/                  # CIFAR-10 dataset
├── models/                # Saved model checkpoints
```

## Requirements

```bash
torch
torchvision
wandb
matplotlib
numpy
scikit-image
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

---

## 3. Literature Review

### Paper 1: Nouveau VAE (NVAE)

**Overview**: NVAE is a deep hierarchical VAE designed to bridge the performance gap between VAEs and other generative models through specialized neural architecture design.

**Architecture Highlights**:
- **Hierarchical Multi-scale Model**: Starts with 8×8 latent space capturing global structure, gradually expanding to pixel level
- **Residual Cells**: Uses depthwise separable convolutions in generative model for larger receptive field
- **Stabilization Techniques**: 
  - Residual Normal Distributions for easier KL divergence optimization
  - Spectral Regularization to prevent unstable gradients
  - Successfully integrates Batch Normalization and Swish activation

**CIFAR-10 Performance**:
- **Score**: 2.91 bpd (Bits Per Dimension) with flow, 2.93 bpd without flow
- **BPD Metric**: Measures average bits needed to encode each pixel. Lower is better. Standard metric for likelihood-based models (VAEs, Normalizing Flows, Autoregressive models).
- **"With Flow"**: Uses Inverse Autoregressive Flows (IAF) in encoder to capture complex dependencies without slowing down sampling.

**Pros & Cons for CIFAR-10**:
- ✅ **Efficient Sampling**: ~8x faster than similar models
- ✅ **Stable Training**: Successfully trained 40+ hierarchical groups
- ❌ **Complexity**: Requires mixed-precision and gradient check-pointing for memory management
- ❌ **Training Time**: Small model took 43-50 hours; full model significantly longer

**Comparison with Other Models**:
| Model | Type | CIFAR-10 (BPD) |
|-------|------|----------------|
| NVAE (w/ flow) | Hierarchical VAE | **2.91** |
| NVAE (w/o flow) | Hierarchical VAE | 2.93 |
| BIVA | Hierarchical VAE | 3.08 |
| IAF-VAE | Hierarchical VAE | 3.11 |
| PixelSNAIL | Autoregressive | 2.85 (slower) |

---

### Paper 2: AASAE (Augmentation-Augmented Stochastic Autoencoders)

**Overview**: AASAE modifies the traditional VAE for Self-Supervised Learning by replacing KL divergence regularization with domain-specific data augmentation to learn better representations for classification tasks.

**Key Innovation**:
- Removes "domain-agnostic" KL term and forces model to reconstruct original images from augmented versions
- "Local packing" strategy makes encoder learn features invariant to image transformations (flips, crops, color changes)
- Still uses stochastic sampling for smooth representation space

**Architecture Highlights**:
- **Encoder**: ResNet-50 backbone + projection layer
- **Decoder**: Inverted ResNet-50 (no Batch Normalization)
- **Augmentation**: SimCLR pipeline (random flips, color drops, resize/crops)

**CIFAR-10 Performance**:
- **Score**: 87.14% downstream classification accuracy (pretrain on 45K images without labels, then train linear classifier)
- **Metric**: Downstream Classification Accuracy - encoder is frozen, single linear layer trained with labels on top

**Pros & Cons for CIFAR-10**:
- ✅ **Massive Improvement**: 30% better than standard VAE (87.14% vs 57.16%)
- ✅ **Efficient Training**: Decomposable loss doesn't require huge batch sizes like SimCLR
- ✅ **Robust**: Stable across different batch sizes and latent dimensions
- ❌ **Not True Generative**: Removing KL term loses probability density estimation capability
- ❌ **Behind SOTA SSL**: Trails top self-supervised methods like SimCLR (94.00%)

**NVAE vs AASAE - Different Goals**:
- **NVAE**: Best for *generating* high-quality images (measures data distribution understanding via BPD)
- **AASAE**: Best for *learning representations* for classification with limited labels (not focused on image generation)

---
