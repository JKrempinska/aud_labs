"""
Environment verification script for Analysis of Unstructured Data (AUD) Lab.
Verifies all required packages, Apple Silicon MPS acceleration, and downloads.
"""
import sys

def check_imports():
    checks = []
    
    # 1. Scientific & Core
    try:
        import numpy as np
        import scipy
        import pandas as pd
        import matplotlib
        import seaborn as sns
        checks.append(("Scientific Core (numpy, scipy, pandas, matplotlib, seaborn)", True, f"numpy {np.__version__}, scipy {scipy.__version__}"))
    except Exception as e:
        checks.append(("Scientific Core", False, str(e)))

    # 2. Scikit-Learn & UMAP
    try:
        import sklearn
        import umap
        checks.append(("Scikit-Learn & UMAP", True, f"sklearn {sklearn.__version__}, umap {umap.__version__}"))
    except Exception as e:
        checks.append(("Scikit-Learn & UMAP", False, str(e)))

    # 3. PyTorch & MPS (Apple Silicon GPU)
    try:
        import torch
        import torchvision
        import torchaudio
        mps_available = torch.backends.mps.is_available()
        mps_str = "MPS (GPU) Available" if mps_available else "MPS Not Available (CPU only)"
        checks.append(("PyTorch stack", True, f"torch {torch.__version__} ({mps_str}), torchvision {torchvision.__version__}, torchaudio {torchaudio.__version__}"))
    except Exception as e:
        checks.append(("PyTorch stack", False, str(e)))

    # 4. NLP (spaCy, NLTK, Gensim, Transformers)
    try:
        import spacy
        import nltk
        import gensim
        import transformers
        nlp = spacy.load("en_core_web_sm")
        checks.append(("NLP stack", True, f"transformers {transformers.__version__}, gensim {gensim.__version__}, spacy {spacy.__version__} (model loaded)"))
    except Exception as e:
        checks.append(("NLP stack", False, str(e)))

    # 5. Computer Vision & Audio
    try:
        import PIL
        import skimage
        import cv2
        import pytorch_grad_cam
        import librosa
        import soundfile
        checks.append(("Vision & Audio stack", True, f"skimage {skimage.__version__}, opencv {cv2.__version__}, librosa {librosa.__version__}"))
    except Exception as e:
        checks.append(("Vision & Audio stack", False, str(e)))

    # 6. FAISS (Vector Indexing)
    try:
        import faiss
        checks.append(("FAISS", True, f"faiss {faiss.__version__}"))
    except Exception as e:
        checks.append(("FAISS", False, str(e)))

    # 7. Jupyter & Widgets
    try:
        import ipykernel
        import ipywidgets
        checks.append(("Jupyter Kernel & Widgets", True, f"ipykernel {ipykernel.__version__}"))
    except Exception as e:
        checks.append(("Jupyter Kernel & Widgets", False, str(e)))

    print("=" * 70)
    print("AUD Lab Environment Verification Report")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print("=" * 70)
    all_passed = True
    for name, status, details in checks:
        sym = "✓" if status else "✗"
        print(f"[{sym}] {name:<35} : {details}")
        if not status:
            all_passed = False
    print("=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED SUCCESSFULLY!")
    else:
        print("SOME CHECKS FAILED.")
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
