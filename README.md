# Mamba-Transformer for Low-light Image Enhancement in HVI Color Space (DBMT)

This repository contains the official implementation of our paper **"Mamba-Transformer for Low-light Image Enhancement in HVI Color Space"**, which proposes a novel framework for enhancing low-light images using a Mamba-Transformer-based backbone and perceptually aligned HVI color space transformation.

---

## 📁 Project Structure

```
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
```

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
```

> ⚠️ **Note**: Installing `causal-conv1d` and `mamba-ssm` may lead to errors if not configured properly. Please follow the official environment setup instructions provided at [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) to ensure correct installation.

---

### 2. Dataset Preparation

Please organize your low-light image dataset according to the configuration specified in `data/options.py`.  
If your dataset structure differs, you may modify the dataset loading paths directly in `data/options.py` to match your setup.

---

### 3. Training

```bash
python train.py
```

---

### 4. Evaluation

```bash
python eval.py
```

---

### 5. Inference Time Benchmark

```bash
python inference_time_test.py
```

---

### 6. Image Quality Metrics

```bash
python measure_niqe_bris.py
```
