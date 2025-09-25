# Mamba-Transformer for Low-light Image Enhancement in HVI Color Space

This repository contains the official implementation of our paper **"Mamba-Transformer for Low-light Image Enhancement in HVI Color Space"**, which proposes a novel framework for enhancing low-light images using a Mamba-Transformer-based backbone and perceptually aligned HVI color space transformation.

## 📁 Project Structure
DBMT/
├── data/                           # Dataset loading and preprocessing
├── loss/                           # Custom loss functions
├── models/                         # Mamba-Transformer architecture
├── eval.py                         # Evaluation script
├── inference_time_test.py          # Inference speed benchmarking
├── measure_niqe_bris.py            # NIQE/BRISQUE quality metric measurement
├── model_quick_test.py             # Model performance evaluation
├── train.py                        # Main training script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

---

## 🚀 Getting Started

### 1. Environment Setup

We recommend using a virtual environment with the following configuration:

- **Python**: 3.11  
- **PyTorch**: 2.4.1  
- **CUDA**: 11.8

Install dependencies:

```bash
pip install -r requirements.txt

### 2. Dataset Preparation
