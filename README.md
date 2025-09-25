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

> ⚠️ Note: Some packages may require manual installation for compatibility with PyTorch 2.4.1. Refer to [PyTorch official site](https://pytorch.org) for platform-specific wheels if needed.

---

### 2. Dataset Preparation

Place your low-light image dataset in the `data/` folder. Supported formats include `.jpg`, `.png`. You may use public datasets like **LOL**, **SID**, or your own custom dataset.

---

### 3. Training

```bash
python train.py
```

Training parameters (e.g., epochs, batch size, learning rate) can be modified in `train.py`.

---

### 4. Evaluation

```bash
python eval.py
```

Evaluates model performance using PSNR, SSIM, and optionally NIQE/BRISQUE.

---

### 5. Inference Time Benchmark

```bash
python inference_time_test.py
```

Reports average inference time per image on your hardware.

---

### 6. Image Quality Metrics

```bash
python measure_niqe_bris.py --input_dir ./data/test_images/
```
