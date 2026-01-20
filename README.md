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
- ✅ 30% better than standard VAE (87.14% vs 57.16%)
- ✅ Decomposable loss doesn't require large batch sizes
- ✅ Stable across different batch sizes and latent dimensions
- ❌ Removing KL term loses probability density estimation capability
- ❌ Lower performance than SimCLR (94.00%)

**NVAE vs AASAE**:
- **NVAE**: Focused on image generation (evaluated via BPD)
- **AASAE**: Focused on representation learning for classification (not focused on generation)

---

### Paper 3: BIVA (Bidirectional-Inference Variational Autoencoder)

**Overview**: BIVA is a deep hierarchical generative model designed to solve "skip-connection" and "posterior collapse" problems in standard deep VAEs by creating bidirectional communication between encoder and decoder.

**Architecture Highlights**:
- **Bidirectional Inference**: Inference process informed by both bottom-up (image features) and top-down (generative path) information, creating an expressive loop
- **Stochastic Skip-Connections**: Passes information from any latent layer directly to output, preventing model from ignoring deep latent variables
- **Dense Hierarchies**: Stacks many "BIVA cells" to model dependencies at multiple scales (global shapes to texture details)

**CIFAR-10 Performance**:
- **Score**: 3.08 BPD (calculated using IWAE sampling for better accuracy)
- **Metric**: Bits Per Dimension (same as NVAE)

**Pros & Cons for CIFAR-10**:
- ✅ Produces sharper images than early VAEs due to deep hierarchy
- ✅ Avoids posterior collapse, uses all latent variables effectively
- ✅ Provides tight bound on data distribution
- ❌ Bidirectional nature makes it slower and more memory-intensive
- ❌ Significantly harder to implement from scratch than standard VAE

**BIVA vs NVAE Comparison**:
| Feature | BIVA (2019) | NVAE (2020) |
|---------|-------------|-------------|
| Main Innovation | Bidirectional Inference | Neural Architecture Design |
| CIFAR-10 Score | 3.08 BPD | **2.91 BPD** |
| Stability | Weight Norm + complex paths | Spectral Regularization |
| Efficiency | Slower (bidirectional loop) | Faster (depthwise convolutions) |
| Image Quality | Good, some noise | Better quality |

NVAE achieves better performance than BIVA by combining hierarchical ideas with modern optimization techniques.

---

### Paper 4: VDVAE (Very Deep VAEs)

**Overview**: VDVAE (OpenAI, 2020) proves that VAEs can match or beat autoregressive models when scaled to extreme depth. The key thesis: previous VAEs underperformed simply because they weren't deep enough. With up to 78 hierarchical levels, it achieved better image compression than almost any other model at the time.

**Architecture Highlights**:
- **Pure Top-Down Hierarchy**: Information flows from small latent space through dozens of layers to final image
- **Residual Learning**: Modified residual cells with Weight Normalization and Layer Scaling (small initialization values) for stability
- **Extreme Depth, Small Width**: 78 layers but only ~39M parameters for CIFAR-10, making it memory-efficient despite depth

**CIFAR-10 Performance**:
- **Score**: 2.87 BPD (calculated using Importance Sampling for tighter, more precise bound)
- **Metric**: Bits Per Dimension (same as NVAE/BIVA)

**Pros & Cons for CIFAR-10**:
- ✅ Best likelihood score (2.87 BPD), beats NVAE and PixelCNN++
- ✅ 70+ scales maintain global structure effectively
- ✅ Generates all pixels at once, unlike autoregressive models
- ❌ Extreme depth prone to posterior collapse without careful tuning
- ❌ Deep hierarchy requires significant compute and long training time
- ❌ 78 stochastic layers require careful gradient management

**VDVAE vs NVAE Comparison**:
| Feature | NVAE (NVIDIA) | VDVAE (OpenAI) |
|---------|---------------|----------------|
| Philosophy | Complex cells with clever design | Simple cells, brute force depth |
| Depth | ~40 hierarchical groups | **78 layers** |
| Parameters | More parameters | ~39M (efficient) |
| Stability Strategy | Spectral Regularization | Layer Scaling + Weight Norm |
| CIFAR-10 Score | 2.91 BPD | **2.87 BPD** |
| Sampling Speed | Very fast | Fast |

**NVAE vs VDVAE**:
- **NVAE**: More efficient with modern architectural components
- **VDVAE**: Achieves better score through extreme depth

---

### Hierarchical VAE Comparison & Model Selection

**Performance & Training Statistics on CIFAR-10**:

| Metric | BIVA (2019) | NVAE (2020) | VDVAE (2020) |
|--------|-------------|-------------|--------------|
| **CIFAR-10 Score (BPD)** | 3.08 | 2.91 | **2.87** |
| **Parameters** | ~25M-100M+ | ~35M | ~39M |
| **Training Speed** | Moderate | **Fastest** | Slowest |
| **Estimated Training Time** | ~2-3 days (V100) | ~40-50 hours (V100) | ~1 week (V100) |

**Architecture Summary**:

**BIVA**:
- ✅ Prevents posterior collapse with skip connections to stochastic layers
- ❌ Bidirectional inference loop difficult to implement from scratch
- ❌ Less efficient than newer specialized convolution types

**NVAE**:
- ✅ Depthwise separable convolutions (lighter and faster)
- ✅ Spectral regularization for KL term stability
- ✅ Production-ready architecture
- ❌ Sensitive to hyperparameter tuning

**VDVAE**:
- ✅ Simple architecture with extreme depth (78 layers)
- ✅ Best BPD score on CIFAR-10 (2.87)
- ❌ Difficult to stabilize (requires specific initialization)
- ❌ Slow training due to sequential overhead

**Selected Model: NVAE**

We selected NVAE as our VAE implementation for the following reasons:

1. **Training Efficiency**: Fastest training among hierarchical VAEs (~40-50 hours vs 1 week for VDVAE)
2. **Balanced Complexity**: Moderate complexity between BIVA and VDVAE
3. **Modern Architecture**: Uses depthwise separable convolutions and spectral regularization
4. **Well-Documented**: Clear ablation studies suitable for replication

---
