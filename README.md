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

---

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


