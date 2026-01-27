# Generative Modeling on CIFAR-10

## Project Overview

This project implements and compares three different generative model architectures on the CIFAR-10 dataset:
- **Variational Autoencoders (VAE)**
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

## 4. Experiment Results

### Debug Experiment 1: NVAE Quick Check
**Goal**: Verify training pipeline, loss convergence, and image generation mechanics.

**Configuration**:
- **Duration**: ~10 minutes
- **Dataset**: 10% of CIFAR-10 (5,000 images)
- **Training**: 5 Epochs
- **Architecture**: Small NVAE (2 scales, 64 hidden dims)

**Observations**:
- ✅ **Pipeline Works**: Data loading, forward/backward pass, and logging are functional.
- ✅ **Loss Decreases**: ELBO loss showed a downward trend even in 5 epochs.
- ⚠️ **Image Quality**: Generated images are currently **blurred** and look visually similar to each other (potential mild posterior collapse common in early VAE training).
- 🔍 **Detail**: Despite the blur, samples show distinguishable **shape features** "inside" the blur, indicating the model is starting to learn structural distributions even with minimal training.

### Debug Experiment 2: NVAE with DMOL Loss
**Goal**: Validate the implementation of the paper-accurate Discretized Mixture of Logistics (DMOL) loss and fix the blurry image issue.

**Configuration**:
- **Duration**: ~10 minutes
- **Dataset**: 10% of CIFAR-10
- **Training**: 10 Epochs
- **Changes**: Switched from MSE/BCE loss to DMOL loss (10 mixture components).

**Observations**:
- ✅ **Metrics**: BPD started around **~5.58**, which is realistic for early training (State-of-the-art is ~2.91).
- ⚠️ **Image Quality**: Unlike the "blurry" output of Experiment 1, the images are now **sharp and colorful**.
- ⚠️ **Noise**: The images are currently **very noisy/pixelated**. This is expected behavior for DMOL loss in early training; the model learns local pixel distributions quickly but needs significantly more training time (50+ epochs) to coordinate them into coherent global structures.
- 🔍 **Conclusion**: The statistical formulation is now correct and matches the NVAE paper. The next step is a full-scale training run to allow convergence.

### Experiment 3: Full NVAE Training
**Goal**: Train the NVAE model for a longer duration to achieve better convergence and evaluate using Importance Weighted Sampling.

**Configuration**:
- **Duration**: ~4 hours (50 Epochs)
- **Dataset**: Full CIFAR-10 (45k Train, 5k Val)
- **Batch Size**: 64
- **Architecture**: NVAE (2 scales, 64 hidden dims, 20 latent dims)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=3e-4) with Cosine Annealing
- **KL Annealing**: 5 warmup epochs

**Results**:
- **Standard ELBO (Test Set)**: 4.59 BPD (Loss: 9780.59)
- **IWELBO (k=100)**: **4.54 BPD** (Loss: 9673.99)
-   *Note*: The Importance Weighted ELBO provides a tighter bound on the true log-likelihood, confirming the model is performing better than the standard training metric suggests.

**Visual Observations**:
- ✅ **Reconstruction**: The model can reconstruct images from the test set with high fidelity, preserving colors and global structure effectively.
- ⚠️ **Generation**: Randomly generated samples (from p(z)) are currently **noisy**. Despite the improved quantitative score (4.54 BPD), the samples lack the sharpness of the reconstructions. This indicates that while the model has learned the data distribution statistics well (low loss), the sampling path might need temperature tuning or more training steps to produce clean images.

### Differences from Official NVAE Implementation
This project implements the core ideas of NVAE but scales them down for feasible training on a single GPU (e.g., Colab T4/V100) within a few hours.

| Feature | Official NVAE Paper | This Implementation |
|---------|---------------------|---------------------|
| **Depth/Scales** | 3 scales with ~30+ hierarchical groups | 2 scales with 3 hierarchical groups (4x4, 8x8, 16x16) |
| **Flows** | Inverse Autoregressive Flows (IAF) for flexible priors | Standard Normal Priors (No Flows) |
| **Training Time** | ~43-50 hours (V100) | ~4 hours (T4/V100) |
| **Batch Size** | Large (e.g., 128 per GPU) | 64 |
| **Precision** | Mixed Precision (FP16/FP32) | Standard FP32 |
| **Parameter Count** | Millions (Deep & Wide) | Lightweight (Hidden Dim 64) |

Despite these simplifications, the model successfully achieves **4.54 BPD**, demonstrating the effectiveness of the NVAE architecture (Residual Cells, Depthwise Separable Convolutions, Spectral Regularization) even at a smaller scale. The official model reaches ~2.91 BPD primarily due to extreme depth and flow-based priors.

### Experiment 4: Extended NVAE Training (100 Epochs)
**Goal**: Investigate if longer training improves sample quality and convergence.

**Configuration**:
- **Duration**: ~8 hours (100 Epochs)
- **Dataset**: Full CIFAR-10
- **Architecture**: Same as Experiment 3

**Results**:
- **Standard ELBO (Test Set)**: 4.47 BPD (Loss: 9524.60)
- **IWELBO (k=100)**: **4.42 BPD** (Loss: 9420.23)
-   *Note*: Continued improvement in quantitative metrics (BPD dropped from 4.54 to 4.42).

**Visual Observations**:
- ⚠️ **Generation Issues Persist**: Despite the improved BPD score, the randomly generated samples remain **noisy** and lack coherent global structure.
- 🔍 **Hypothesis**: The model might be optimizing for pixel-level statistics (low loss) without capturing the global data manifold effectively, or the sampling temperature/procedure needs further adjustment. The disconnect between good BPD and poor samples is a known phenomenon in likelihood-based models.

## 5. Current Architecture Details

> **Deep Dive Available**: For a comprehensive, line-by-line explanation of the NVAE architecture, math, and code components, please read [NVAE_ARCHITECTURE_DETAILS.md](./NVAE_ARCHITECTURE_DETAILS.md).

The implemented model is a lightweight version of the **Nouveau VAE (NVAE)**, adapted for efficient training on CIFAR-10 while retaining key architectural innovations.

### Core Components
- **Activation Function**: Swish ($x \cdot \sigma(x)$) is used throughout the network instead of ReLU for better gradient flow.
- **Residual Cells**: The fundamental building block uses **Depthwise Separable Convolutions** to increase receptive fields efficiently.
  - Structure: $1\times1$ Conv (Expansion) $\to$ $5\times5$ Depthwise Conv $\to$ $1\times1$ Conv (Projection) $\to$ Squeeze-and-Excitation (SE) Block.
  - **Spectral Normalization**: Applied to convolution layers to stabilize training.

### Encoder (Bottom-Up)
The encoder extracts hierarchical features from the input image ($32\times32\times3$).
1. **Stem**: Initial convolution mapping input to hidden dimensions.
2. **Hierarchical Stages**:
   - Stage 1: Processes $32\times32$ features.
   - Stage 2: Downsamples to $16\times16$.
   - Stage 3: Downsamples to $8\times8$.
   - Stage 4: Downsamples to $4\times4$.
   - Features from each stage are stored for skip connections to the decoder.

### Decoder (Top-Down)
The decoder generates the image by processing latent variables from coarse to fine scales.
1. **Input**: Starts with a learnable constant parameter at $4\times4$ resolution.
2. **Hierarchical Latent Variables**:
   - **Scale 1 ($4\times4$)**: Coarse global structure.
   - **Scale 2 ($8\times8$)**: Mid-level details.
   - **Scale 3 ($16\times16$)**: Fine details.
   - At each scale, the posterior $q(z|x)$ combines top-down decoder features with bottom-up encoder features (via residual connections).
   - The prior $p(z)$ is learned from the top-down features alone.
3. **Upsampling**: Uses bilinear interpolation followed by convolution.

### Output & Loss Function
- **Output Layer**: The final $32\times32$ feature map is projected to 100 channels to parameterize a **Discretized Mixture of Logistics (DMOL)** distribution (10 mixtures).
- **Reconstruction Loss**: Negative Log-Likelihood of the DMOL distribution. This models the multimodal nature of pixel values better than MSE or BCE.
- **KL Divergence**: Calculated analytically for Gaussian distributions. Summed over all scales.

### Experiment 5: Continued NVAE Training (125 Epochs)
**Goal**: Further extend training to improve sample quality and convergence, continuing from the 100-epoch checkpoint.

**Configuration**:
- **Duration**: Extended to 125 Epochs total.
- **Warmup Annealing**: Increased to 110 epochs.
  - *Strategy*: By setting the warmup (KL annealing) duration to 110 epochs (just past the resume point of 100), the training process re-introduced a short annealing phase. This allowed the model to temporarily relax the KL constraint and explore the latent space more freely before tightening the bound again, a technique often used to help models escape local minima during fine-tuning.

**Results**:
- **Standard ELBO (Validation Set)**: 4.46 BPD (Loss: ~9489.09 at Epoch 121)
-   *Note*: The validation loss reached its minimum around epoch 121, showing a slight improvement over the 100-epoch result (previous ~4.47-4.59 BPD range).

**Visual Observations**:
- ⚠️ **Noise Persists**: The generated samples remain noisy.
- 🔍 **Conclusion**: While the model is mathematically improving (lower NLL/BPD), the perceptual quality of *random samples* has not yet converged to clean images. This suggests that either significantly more training time is required for the pixel-level distributions to align globally, or the model capacity (depth) needs to be increased to match the official NVAE results.

### Experiment 6: Continued Training to 200 Epochs
**Goal**: Push the training duration significantly further to see if the model can break through the "noisy sample" barrier.

**Configuration**:
- **Duration**: Extended to 200 Epochs total (resuming from 150).
- **Warmup Annealing**: Adjusted to 165 epochs to provide another relaxation phase.

**Results**:
- **Standard ELBO (Test Set)**: 4.4301
-   *Note*: A clear quantitative improvement, dropping below 4.44 BPD for the first time.

**Visual Observations**:
- 🔍 **Emerging Structure**: While samples are still noisy, there is a visible improvement in coherence. Forms and shapes are starting to emerge more clearly from the noise compared to earlier epochs, suggesting the model is beginning to grasp better global structure, though it still requires more training or capacity to fully resolve clean images.
