# Facial Expression Recognition System

## Overview  
This solution delivers a production-ready deep learning pipeline for facial expression classification across seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. The workflow operationalizes EfficientNet-B0 (via TIMM), advanced augmentation strategies, and GPU-optimized training loops to ensure scalable, repeatable, and high-performance model development for real-world image-based emotion recognition.

## Key Features  
- Production-aligned dataset ingestion using `ImageFolder` and PyTorch `DataLoader`  
- Robust augmentation pipeline powered by torchvision transforms  
- EfficientNet-B0 backbone with transfer-learning optimization  
- Comprehensive training–validation loop with loss/accuracy telemetry  
- Automated checkpointing for best-performing weights  
- Inference-ready preprocessing (grayscale support, normalization, tensor conversion)  
- Visualization utilities for probability distribution and interpretability  

## Dataset Requirement  
The dataset must follow an `ImageFolder` directory structure, where the root contains two primary folders: `train` and `val`. Each of these folders must include seven subdirectories corresponding to the facial expression classes: **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**. Every subdirectory should contain its respective image samples, ensuring a consistent layout for model training and validation. 

## Workflow Architecture  
1. Clone dataset and configure workspace  
2. Build PyTorch dataloaders using `ImageFolder`  
3. Apply augmentation pipeline (resize, normalize, flips, grayscale compliance)  
4. Load EfficientNet-B0 through TIMM and prepare classifier head  
5. Execute training loop with accuracy/loss tracking  
6. Validate per epoch and store best-weight checkpoints  
7. Run inference utilities for predictions and probability visualization  

## Installation  

### 1. Clone the repository 
```bash
git clone <your-repo-url>
cd <project-folder>

