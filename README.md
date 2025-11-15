# Facial Expression Recognition System

A Production-Ready Deep Learning Workflow
This project operationalizes an end-to-end facial expression recognition system built on PyTorch, Timm, and EfficientNet-B0. The solution integrates advanced augmentation strategies, streamlined data loaders, automated training loops, and model-driven inference utilities. The objective is to deliver a scalable, repeatable, and high-performance pipeline for image-based emotion classification.

---

## Overview  
This repository enables rapid model development for seven key facial-expression classes:

- **angry**
- **disgust**
- **fear**
- **happy**
- **neutral**
- **sad**
- **surprise**

The workflow leverages **EfficientNet-B0**, pre-trained on ImageNet, and fine-tuned on the target dataset using modern training heuristics. The architecture is optimized for extensibility and can be deployed across GPU-accelerated environments.

---

## Key Features  
- **Production-aligned data ingestion** using `ImageFolder` and PyTorch `DataLoader`  
- **Robust augmentation pipeline** powered by torchvision transforms  
- **Transfer-learning optimized model** via TIMM's EfficientNet-B0 backbone  
- **Comprehensive train–validate loop** with loss tracking and accuracy monitoring  
- **Automated checkpointing** of best-performing weights  
- **Inference-ready image preprocessing** with grayscale conversion and normalization  
- **Insightful visualization utility** for class-probability breakdowns  

---

## Environment Setup  

Ensure all dependencies are provisioned using the following installs:

```bash
git clone https://github.com/parth1620/Facial-Expression-Dataset.git
pip install -U git+https://github.com/albumentations-team/albumentations
pip install timm
pip install --upgrade opencv-contrib-python
