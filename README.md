# Analysis of Unstructured Data (AUD) - Lab Environment

This repository contains the environment setup, starter notebooks, and resources for the **Analysis of Unstructured Data** lab assignments (Janusz Szwabiński).

---

## 🚀 Quick Start

### 1. Activate the Virtual Environment

The virtual environment has been created with Python 3.12:

```bash
# In this directory:
source .venv/bin/activate
```

### 2. Verify the Environment

You can verify that all packages, models, and hardware acceleration (Apple Silicon MPS) are ready:

```bash
python check_env.py
```

### 3. Open Notebooks & Select Kernel

You can work directly in VSCode / Antigravity IDE or start a Jupyter Notebook server:

```bash
jupyter lab
# or
jupyter notebook
```

> **Kernel:** When opening any notebook, select the registered kernel: **`AUD Lab (Python 3.12)`** (internal name: `aoud-lab`).

---

## 📁 Repository Structure

```text
lab/
├── AUD_Lab_assignments.pdf    # Original course assignments specification
├── requirements.txt           # Frozen package dependencies
├── check_env.py               # Environment validation script
├── generate_notebooks.py      # Notebook generator script
├── .gitignore                 # Configured for models, venvs, and cache
│
├── problem_set_1/
│   └── problem_set_1.ipynb    # PS1: Vector representations, similarity & PCA/t-SNE/UMAP
├── problem_set_2/
│   └── problem_set_2.ipynb    # PS2: Neural networks & backprop, NumPy & PyTorch autograd
├── problem_set_3/
│   └── problem_set_3.ipynb    # PS3: Text representations (TF-IDF, Naïve Bayes, Word2Vec, GloVe)
├── problem_set_4/
│   └── problem_set_4.ipynb    # PS4: Attention & Transformers (MHA from scratch, HuggingFace)
├── problem_set_5/
│   └── problem_set_5.ipynb    # PS5: Images (2-D Conv, Sobel, HOG, CNN with augmentation)
├── problem_set_6/
│   └── problem_set_6.ipynb    # PS6: Vision Transfer Learning & Grad-CAM (ResNet, ViT)
├── problem_set_7/
│   └── problem_set_7.ipynb    # PS7: Audio processing (FFT, Spectrogram, MFCCs, CNN classifier)
└── problem_set_8/
    └── problem_set_8.ipynb    # PS8: Multimodal retrieval, Autoencoders & FAISS vector search
```

---

## 📦 Installed Libraries & Tooling

| Component | Libraries |
| :--- | :--- |
| **Core Scientific** | `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn` |
| **Classical ML & Manifolds** | `scikit-learn`, `umap-learn` |
| **Deep Learning** | `torch`, `torchvision`, `torchaudio` (Apple Silicon MPS accelerated) |
| **NLP & Transformers** | `transformers`, `datasets`, `accelerate`, `spacy` (`en_core_web_sm`), `nltk`, `gensim` |
| **Computer Vision** | `scikit-image`, `opencv-python`, `pillow`, `grad-cam` |
| **Audio Processing** | `librosa`, `soundfile` |
| **Scalable Vector Search**| `faiss-cpu` |
| **Notebook Runtime** | `jupyter`, `ipykernel`, `ipywidgets`, `tqdm` |

---

## ⚡ Hardware Acceleration (Apple Silicon MPS)

In PyTorch coding problems, you can leverage GPU acceleration on Apple Silicon:

```python
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```
