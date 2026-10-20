import json
import os

NOTEBOOK_KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "AUD Lab (Python 3.12)",
        "language": "python",
        "name": "aoud-lab"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.14"
    }
}

problem_sets = [
    {
        "id": "problem_set_1",
        "num": 1,
        "title": "Problem Set 1. Vector representations, similarity & dimensionality reduction",
        "goal": "Establish the “everything is a vector/tensor” paradigm and the geometry it lives in.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "Let $x, y \\in \\mathbb{R}^d$. Show that if $\\|x\\|_2 = \\|y\\|_2 = 1$ then $\\|x - y\\|_2^2 = 2 (1 - \\cos \\theta)$, where $\\cos \\theta$ is the cosine similarity. Deduce that on the unit sphere Euclidean and cosine geometry induce the same ranking of neighbours.",
                "type": "markdown",
                "content": "### Problem 1 [Theory]\n\n**Task:**\nLet $x, y \\in \\mathbb{R}^d$. Show that if $\\|x\\|_2 = \\|y\\|_2 = 1$ then $\\|x - y\\|_2^2 = 2 (1 - \\cos \\theta)$, where $\\cos \\theta$ is the cosine similarity. Deduce that on the unit sphere Euclidean and cosine geometry induce the same ranking of neighbours.\n\n---\n**Derivation & Solution:**\n\n*(Write your derivation here)*\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "Define the cosine distance $d_{\\cos}(x, y) = 1 - \\cos \\theta(x, y)$. Give an explicit counterexample in $\\mathbb{R}^2$ showing that $d_{\\cos}$ violates the triangle inequality, hence is not a metric. Contrast with the angular distance $\\arccos(\\cos \\theta)/\\pi$.",
                "type": "markdown",
                "content": "### Problem 2 [Theory]\n\n**Task:**\nDefine the cosine distance $d_{\\cos}(x, y) = 1 - \\cos \\theta(x, y)$. Give an explicit counterexample in $\\mathbb{R}^2$ showing that $d_{\\cos}$ violates the triangle inequality, hence is not a metric. Contrast with the angular distance $\\arccos(\\cos \\theta)/\\pi$.\n\n---\n**Counterexample & Discussion:**\n\n*(Provide three vectors in $\\mathbb{R}^2$ and verify the violation of the triangle inequality)*\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "(Curse of dimensionality.) For $n$ points drawn i.i.d. uniformly in $[0, 1]^d$, argue that the relative gap between the nearest and farthest point from a fixed query, $(D_{\\max}-D_{\\min})/D_{\\min}$, tends to 0 as $d \\to \\infty$. Separately, show the fraction of the volume of a $d$-ball within an $\\varepsilon$-shell of its surface, $1 - (1 - \\varepsilon)^d$, tends to 1. Comment on the consequences for nearest-neighbour search.",
                "type": "markdown",
                "content": "### Problem 3 [Theory] (Curse of dimensionality)\n\n**Task:**\nFor $n$ points drawn i.i.d. uniformly in $[0, 1]^d$, argue that the relative gap between the nearest and farthest point from a fixed query, $(D_{\\max}-D_{\\min})/D_{\\min}$, tends to 0 as $d \\to \\infty$. Separately, show the fraction of the volume of a $d$-ball within an $\\varepsilon$-shell of its surface, $1 - (1 - \\varepsilon)^d$, tends to 1. Comment on the consequences for nearest-neighbour search.\n\n---\n**Derivation & Discussion:**\n\n*(Write your mathematical argument and discussion here)*\n"
            },
            {
                "tag": "[Theory]",
                "num": 4,
                "text": "(PCA three ways.) Let $X \\in \\mathbb{R}^{n \\times p}$ be column-centred with SVD $X = U\\Sigma V^\\top$. Prove that the direction $w$ (with $\\|w\\| = 1$) maximising the sample variance $\\mathrm{Var}(Xw)$ is the top right singular vector $v_1$, and that this coincides with the leading eigenvector of the covariance $\\frac{1}{n} X^\\top X$. Show further that the rank-$k$ PCA projection minimises the reconstruction error $\\sum_{i} \\|x_i - \\hat{x}_i\\|_2^2$.",
                "type": "markdown",
                "content": "### Problem 4 [Theory] (PCA three ways)\n\n**Task:**\nLet $X \\in \\mathbb{R}^{n \\times p}$ be column-centred with SVD $X = U\\Sigma V^\\top$. Prove that the direction $w$ (with $\\|w\\| = 1$) maximising the sample variance $\\mathrm{Var}(Xw)$ is the top right singular vector $v_1$, and that this coincides with the leading eigenvector of the covariance $\\frac{1}{n} X^\\top X$. Show further that the rank-$k$ PCA projection minimises the reconstruction error $\\sum_{i} \\|x_i - \\hat{x}_i\\|_2^2$.\n\n---\n**Proof:**\n\n*(Write your proof here)*\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Load one short text document, one image and one audio clip and turn each into a NumPy array. Report the shape and dtype of each, and write two sentences per modality explaining what a “feature vector” would mean in each case.",
                "type": "code",
                "code": "# Problem 5 [Python]\nimport numpy as np\nfrom PIL import Image\nimport librosa\n\n# 1. Text -> array representation\n# 2. Image -> array representation\n# 3. Audio -> array representation\n\n# Report shape and dtype for each, and explain what feature vectors represent\n"
            },
            {
                "tag": "[Python]",
                "num": 6,
                "text": "Implement PCA from scratch via the SVD and reproduce sklearn.decomposition.PCA on the digits dataset (agree to 10^-6 on the explained-variance ratios). Plot the cumulative explained variance and the 2-D projection coloured by class.",
                "type": "code",
                "code": "# Problem 6 [Python]\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\n\n# Load digits dataset\ndigits = load_digits()\nX, y = digits.data, digits.target\n\n# TODO: Implement PCA from scratch via SVD and verify against sklearn.decomposition.PCA\n"
            },
            {
                "tag": "[Python]",
                "num": 7,
                "text": "Run t-SNE and UMAP on the same features. Produce the three 2-D embeddings side by side (PCA, t-SNE, UMAP).",
                "type": "code",
                "code": "# Problem 7 [Python]\nfrom sklearn.manifold import TSNE\nimport umap\nimport matplotlib.pyplot as plt\n\n# TODO: Compute 2D embeddings with PCA, t-SNE, and UMAP and plot side-by-side\n"
            },
            {
                "tag": "[Analysis]",
                "num": 8,
                "text": "Compare the three embeddings from Problem 7: which one preserves global distances, which preserves local neighbourhoods, and why can t-SNE/UMAP distances not be read as similarities? Relate your answer to Problems 1–3.",
                "type": "markdown",
                "content": "### Problem 8 [Analysis]\n\n**Task:**\nCompare the three embeddings from Problem 7: which one preserves global distances, which preserves local neighbourhoods, and why can t-SNE/UMAP distances not be read as similarities? Relate your answer to Problems 1–3.\n\n---\n**Analysis & Discussion:**\n\n*(Write your analysis here)*\n"
            }
        ]
    },
    {
        "id": "problem_set_2",
        "num": 2,
        "title": "Problem Set 2. Neural networks & optimisation",
        "goal": "Derive the deep-learning backbone once, so it can be reused for text, images and audio.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "(Softmax + cross-entropy.) For logits $z \\in \\mathbb{R}^K$, $\\mathrm{softmax}(z)_k = p_k = e^{z_k} / \\sum_j e^{z_j}$ and one-hot target $y$, show that the cross-entropy loss $L = -\\sum_k y_k \\log p_k$ has gradient $\\partial L / \\partial z = p - y$. Explain why this clean form is the reason softmax and cross-entropy are always paired.",
                "type": "markdown",
                "content": "### Problem 1 [Theory] (Softmax + cross-entropy)\n\n---\n**Derivation & Explanation:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "(Backpropagation.) Consider a two-layer MLP $h = \\sigma(W_1 x + b_1)$, $\\hat{y} = \\mathrm{softmax}(W_2 h + b_2)$ with cross-entropy loss. Derive $\\partial L/\\partial W_2, \\partial L/\\partial b_2, \\partial L/\\partial W_1, \\partial L/\\partial b_1$ explicitly using the chain rule, for a general elementwise activation $\\sigma$.",
                "type": "markdown",
                "content": "### Problem 2 [Theory] (Backpropagation)\n\n---\n**Derivation:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "Write the update rules of SGD, SGD+momentum and Adam. Show that momentum is an exponential moving average of gradients, and explain the purpose of Adam’s bias-correction terms $\\hat{m}_t, \\hat{v}_t$ in the first few steps.",
                "type": "markdown",
                "content": "### Problem 3 [Theory] (Optimisers)\n\n---\n**Update Rules & Analysis:**\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Implement the two-layer MLP of Problem 2 in pure NumPy, including the backward pass. Verify every gradient by central finite differences (gradient check to < 10^-5 relative error).",
                "type": "code",
                "code": "# Problem 4 [Python]\nimport numpy as np\n\n# TODO: Implement 2-layer MLP in pure NumPy with backpropagation and gradient check\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Re-implement the same network in PyTorch with autograd. Train it on the Problem-Set-1 feature set and reproduce the NumPy loss curve to confirm the implementations agree.",
                "type": "code",
                "code": "# Problem 5 [Python]\nimport torch\nimport torch.nn as nn\nimport matplotlib.pyplot as plt\n\n# TODO: PyTorch autograd implementation matching NumPy MLP\n"
            },
            {
                "tag": "[Python]",
                "num": 6,
                "text": "Study regularisation: train with/without L2, dropout and early stopping; produce train/validation learning curves for each and a small table of final accuracies.",
                "type": "code",
                "code": "# Problem 6 [Python]\n# TODO: Study regularisation (L2, dropout, early stopping) and plot curves\n"
            },
            {
                "tag": "[Analysis]",
                "num": 7,
                "text": "Using the curves from Problem 6, identify where over- and under-fitting occur and argue which regulariser helped most on this dataset and why.",
                "type": "markdown",
                "content": "### Problem 7 [Analysis]\n\n---\n**Discussion:**\n"
            }
        ]
    },
    {
        "id": "problem_set_3",
        "num": 3,
        "title": "Problem Set 3. Text I/II: classical representations, classification & embeddings",
        "goal": "Move from sparse count vectors to dense distributional semantics.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "Derive the TF-IDF weight $w_{t,d} = \\mathrm{tf}_{t,d} \\cdot \\log \\frac{N}{\\mathrm{df}_t}$. Give an information-theoretic reading of the $\\log(N/\\mathrm{df}_t)$ term and explain why raw term frequency alone is a poor weighting.",
                "type": "markdown",
                "content": "### Problem 1 [Theory] (TF-IDF)\n\n---\n**Derivation & Explanation:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "(Multinomial Naïve Bayes.) Derive the MAP parameter estimates with Laplace smoothing, $\\hat{\\theta}_{t|c} = \\frac{n_{t,c} + \\alpha}{\\sum_{t'} n_{t',c} + \\alpha |V|}$. Show that the resulting decision rule is linear in the word-count vector, and compare its form to logistic regression.",
                "type": "markdown",
                "content": "### Problem 2 [Theory] (Multinomial Naïve Bayes)\n\n---\n**Derivation & Comparison:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "(Skip-gram with negative sampling.) Write the SGNS objective $\\log \\sigma(v_c^\\top v_w) + \\sum_{i=1}^k \\mathbb{E}_{w_i \\sim P_n} \\log \\sigma(-v_{w_i}^\\top v_w)$. Derive the gradient w.r.t. a centre-word vector $v_w$, and explain informally why SGNS approximates a shifted PMI factorisation.",
                "type": "markdown",
                "content": "### Problem 3 [Theory] (Skip-gram with negative sampling)\n\n---\n**Objective, Gradient & PMI Factorisation:**\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Build a TF-IDF pipeline (spaCy/NLTK preprocessing -> TfidfVectorizer) and train both multinomial Naïve Bayes and logistic regression on a text-classification dataset (e.g. 20 Newsgroups subset). Report precision/recall/F1 and list the most informative features per class.",
                "type": "code",
                "code": "# Problem 4 [Python]\nimport spacy\nimport nltk\nfrom sklearn.datasets import fetch_20newsgroups\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report\n\n# TODO: Implement TF-IDF pipeline, train MNB & LogReg, report metrics and top features\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Train word2vec with gensim on the same corpus and separately load pretrained GloVe. For both, return the nearest neighbours of five query words and evaluate on an analogy set (e.g. king - man + woman).",
                "type": "code",
                "code": "# Problem 5 [Python]\nimport gensim\nfrom gensim.models import Word2Vec\nimport gensim.downloader as api\n\n# TODO: Train Word2Vec and evaluate queries + analogies\n"
            },
            {
                "tag": "[Analysis]",
                "num": 6,
                "text": "Probe the embeddings for social bias (e.g. gendered analogies or association tests). Report what you find in one paragraph and comment on the risk of propagating such bias downstream.",
                "type": "markdown",
                "content": "### Problem 6 [Analysis] (Embedding bias)\n\n---\n**Findings & Discussion:**\n"
            }
        ]
    },
    {
        "id": "problem_set_4",
        "num": 4,
        "title": "Problem Set 4. Text III: attention & transformers",
        "goal": "Understand and reproduce the attention mechanism, then use a pretrained transformer.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "Derive scaled dot-product attention $\\mathrm{Attn}(Q, K, V) = \\mathrm{softmax}(QK^\\top / \\sqrt{d_k}) V$. Give the variance argument for the $1/\\sqrt{d_k}$ scaling (what happens to the softmax as $d_k$ grows without it?).",
                "type": "markdown",
                "content": "### Problem 1 [Theory] (Scaled dot-product attention)\n\n---\n**Derivation & Scaling Argument:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "Show that self-attention is permutation-equivariant, i.e. permuting the input rows permutes the output rows identically. Explain why this forces the use of positional encodings, and analyse the sinusoidal encoding’s ability to represent relative positions.",
                "type": "markdown",
                "content": "### Problem 2 [Theory] (Permutation equivariance & Positional Encodings)\n\n---\n**Proof & Analysis:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "Count the parameters and the asymptotic compute of one multi-head attention layer with $h$ heads and model dimension $d$, and compare with a dense layer of the same width. Justify the $O(n^2 d)$ cost in sequence length $n$.",
                "type": "markdown",
                "content": "### Problem 3 [Theory] (Parameter count and asymptotic compute)\n\n---\n**Computation & Comparison:**\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Implement scaled dot-product attention and multi-head attention from scratch and check your output against torch.nn.MultiheadAttention on random inputs (max abs. error < 10^-5).",
                "type": "code",
                "code": "# Problem 4 [Python]\nimport torch\nimport torch.nn as nn\n\n# TODO: Implement scaled dot-product & MultiheadAttention from scratch and verify against torch.nn.MultiheadAttention\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Use a pretrained HuggingFace transformer two ways: (a) as a frozen sentence-embedding extractor feeding a linear classifier, and (b) fine-tuned end-to-end. Compare both against the TF-IDF baseline from Problem Set 3 on the same data.",
                "type": "code",
                "code": "# Problem 5 [Python]\nfrom transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification\nimport torch\n\n# TODO: (a) Frozen sentence-embedding extractor + linear classifier\n#       (b) Fine-tuned end-to-end\n#       Compare with PS3 TF-IDF\n"
            },
            {
                "tag": "[Analysis]",
                "num": 6,
                "text": "Visualise the attention weights of one head for two example sentences and discuss whether the pattern is interpretable. Note at least one caveat about reading meaning into attention maps.",
                "type": "markdown",
                "content": "### Problem 6 [Analysis]\n\n---\n**Visualisation & Interpretation:**\n"
            }
        ]
    },
    {
        "id": "problem_set_5",
        "num": 5,
        "title": "Problem Set 5. Images I/II: convolution, classical features & CNNs",
        "goal": "Connect hand-crafted image features to learned convolutional features.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "Give the discrete definitions of convolution and cross-correlation for 2-D signals and prove (i) linearity and (ii) shift-equivariance. State precisely how the two operations differ and why deep-learning “convolution” is really cross-correlation.",
                "type": "markdown",
                "content": "### Problem 1 [Theory]\n\n---\n**Definitions & Proofs:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "Derive the output-size formula for a conv layer with input size $W$, kernel $k$, padding $p$, stride $s$ and dilation $\\delta$. Then count the trainable parameters of a conv layer with $C_{\\mathrm{in}}$ input and $C_{\\mathrm{out}}$ output channels, and compare with the equivalent dense layer.",
                "type": "markdown",
                "content": "### Problem 2 [Theory]\n\n---\n**Derivation & Parameter Count:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "Describe the Sobel operator and show how HOG builds a descriptor from histograms of gradient orientations. Which geometric/photometric invariances does HOG have, and which does it lack?",
                "type": "markdown",
                "content": "### Problem 3 [Theory]\n\n---\n**Sobel, HOG and Invariances:**\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Implement 2-D convolution from scratch (valid and “same” padding) and verify it against scipy.signal.convolve2d. Apply a blur and the Sobel edge detector to a sample image.",
                "type": "code",
                "code": "# Problem 4 [Python]\nimport numpy as np\nfrom scipy import signal\nimport matplotlib.pyplot as plt\n\n# TODO: Implement 2D convolution from scratch, compare with scipy.signal.convolve2d, apply blur & Sobel\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Compute HOG features (scikit-image) for a small image dataset and train an SVM / logistic classifier on them; record the accuracy as the classical baseline.",
                "type": "code",
                "code": "# Problem 5 [Python]\nfrom skimage.feature import hog\nfrom sklearn.svm import LinearSVC\nfrom torchvision.datasets import FashionMNIST\n\n# TODO: Extract HOG features, train classifier, record accuracy\n"
            },
            {
                "tag": "[Python]",
                "num": 6,
                "text": "Train a CNN in PyTorch on Fashion-MNIST or CIFAR-10 with data augmentation. Report the confusion matrix and compare accuracy and training cost against the HOG baseline from Problem 5.",
                "type": "code",
                "code": "# Problem 6 [Python]\nimport torch\nimport torch.nn as nn\nimport torchvision\nimport torchvision.transforms as transforms\n\n# TODO: Train CNN with data augmentation, evaluate confusion matrix & compare with HOG\n"
            },
            {
                "tag": "[Analysis]",
                "num": 7,
                "text": "Perform an error analysis on the CNN: show the most confidently wrong examples and the classes most often confused. Quantify the effect of augmentation with an ablation.",
                "type": "markdown",
                "content": "### Problem 7 [Analysis]\n\n---\n**Error Analysis & Ablation:**\n"
            }
        ]
    },
    {
        "id": "problem_set_6",
        "num": 6,
        "title": "Problem Set 6. Images III: transfer learning & modern architectures",
        "goal": "Reuse pretrained representations and understand what makes deep vision nets trainable.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "(Residual connections.) For a residual block $y = x + F(x)$, compute $\\partial y / \\partial x$ and explain, via the chain rule across many blocks, how the identity path mitigates vanishing gradients compared with a plain stack $y = F(x)$.",
                "type": "markdown",
                "content": "### Problem 1 [Theory] (Residual connections)\n\n---\n**Derivation & Analysis:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "(Batch normalisation.) Write the train-time normalisation and the learnable $\\gamma, \\beta$, and state precisely how the eval-time statistics differ. Give one reason BN can accelerate optimisation.",
                "type": "markdown",
                "content": "### Problem 2 [Theory] (Batch normalisation)\n\n---\n**Derivation & Explanation:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "Show that a Vision Transformer’s patch-embedding step is a strided convolution, and connect the rest of the ViT to the attention derivation of Problem Set 4. Contrast the inductive biases of CNNs and ViTs and predict which needs more data.",
                "type": "markdown",
                "content": "### Problem 3 [Theory] (ViT vs CNN)\n\n---\n**Analysis & Comparison:**\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Fine-tune a pretrained torchvision ResNet on a small custom dataset. Compare two regimes: (a) frozen backbone + trained head (feature extraction) and (b) full fine-tuning. Plot a data-efficiency curve (accuracy vs. number of training images).",
                "type": "code",
                "code": "# Problem 4 [Python]\nimport torch\nimport torchvision.models as models\nimport matplotlib.pyplot as plt\n\n# TODO: Compare frozen backbone vs full fine-tuning; plot data-efficiency curve\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Produce Grad-CAM (or a comparable saliency) visualisation for several test images and check whether the network attends to sensible regions.",
                "type": "code",
                "code": "# Problem 5 [Python]\nfrom pytorch_grad_cam import GradCAM\nfrom pytorch_grad_cam.utils.image import show_cam_on_image\n\n# TODO: Produce Grad-CAM saliency maps on test images\n"
            },
            {
                "tag": "[Analysis]",
                "num": 6,
                "text": "Discuss when transfer learning helps and when it hurts, using your results and the idea of domain shift. Give one concrete situation where fine-tuning the whole backbone would be a mistake.",
                "type": "markdown",
                "content": "### Problem 6 [Analysis]\n\n---\n**Discussion on Transfer Learning & Domain Shift:**\n"
            }
        ]
    },
    {
        "id": "problem_set_7",
        "num": 7,
        "title": "Problem Set 7. Audio: signals, spectral features & machine learning",
        "goal": "Turn a waveform into features and then classify it.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "State the sampling theorem and define the Nyquist frequency. Derive the DFT $X_k = \\sum_{n=0}^{N-1} x_n e^{-2\\pi i kn / N}$ and sketch the divide-and-conquer argument giving the FFT’s $O(N \\log N)$ cost. Explain aliasing in one sentence.",
                "type": "markdown",
                "content": "### Problem 1 [Theory] (Sampling theorem, DFT & FFT)\n\n---\n**Derivation & Explanation:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "Define the STFT and the spectrogram and state the time–frequency resolution trade-off controlled by the window length. Then write the MFCC pipeline (mel filterbank -> log -> DCT) and explain what role the DCT plays (decorrelation / energy compaction).",
                "type": "markdown",
                "content": "### Problem 2 [Theory] (STFT & MFCC)\n\n---\n**Derivation & Pipeline Explanation:**\n"
            },
            {
                "tag": "[Python]",
                "num": 3,
                "text": "With librosa, load an audio clip and produce: the waveform, the magnitude spectrum (FFT), the spectrogram, the mel-spectrogram and MFCCs, each as a labelled figure. Demonstrate aliasing by deliberately down-sampling below Nyquist and describing the artefact.",
                "type": "code",
                "code": "# Problem 3 [Python]\nimport librosa\nimport librosa.display\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# TODO: Produce waveform, FFT magnitude spectrum, STFT spectrogram, mel-spectrogram, MFCCs\n#       Demonstrate downsampling aliasing artefact\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Train two classifiers on a small audio dataset (e.g. Speech Commands or an ESC-50 subset): (a) MFCC features + a classical model, and (b) a log-mel spectrogram + a small CNN (the “spectrogram-as-image” approach). Compare accuracy and discuss.",
                "type": "code",
                "code": "# Problem 4 [Python]\n# TODO: Train (a) MFCC + classical classifier and (b) log-mel + small CNN\n"
            },
            {
                "tag": "[Analysis]",
                "num": 5,
                "text": "Explain in one page the alignment problem in sequence labelling and the intuition behind CTC, and contrast keyword spotting with full ASR in terms of what each model must output.",
                "type": "markdown",
                "content": "### Problem 5 [Analysis]\n\n---\n**Sequence Labelling, CTC & ASR:**\n"
            }
        ]
    },
    {
        "id": "problem_set_8",
        "num": 8,
        "title": "Problem Set 8. Cross-cutting: self-supervised, multimodal & scalable retrieval",
        "goal": "Representation learning without labels, shared embedding spaces, and search at scale.",
        "problems": [
            {
                "tag": "[Theory]",
                "num": 1,
                "text": "(Linear autoencoder ≡ PCA.) For a linear autoencoder with tied weights and squared reconstruction loss, show that the optimal encoder spans the same subspace as the top principal components. State where the equivalence breaks once nonlinearities are added.",
                "type": "markdown",
                "content": "### Problem 1 [Theory] (Linear Autoencoder == PCA)\n\n---\n**Derivation & Analysis:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 2,
                "text": "(Contrastive learning.) Write the InfoNCE loss for a positive pair among $N$ candidates, derive its gradient w.r.t. the query embedding, and explain the role of the temperature $\\tau$. State in what sense InfoNCE lower-bounds mutual information.",
                "type": "markdown",
                "content": "### Problem 2 [Theory] (Contrastive Learning & InfoNCE)\n\n---\n**Derivation & Lower Bound:**\n"
            },
            {
                "tag": "[Theory]",
                "num": 3,
                "text": "(Scalable search.) For signed feature hashing, show that the hashed inner product is an unbiased estimator of the true inner product and give its variance’s dependence on the number of buckets. For cosine LSH with random hyperplanes, show the collision probability is $1 - \\theta/\\pi$.",
                "type": "markdown",
                "content": "### Problem 3 [Theory] (Scalable Search & LSH)\n\n---\n**Proofs:**\n"
            },
            {
                "tag": "[Python]",
                "num": 4,
                "text": "Train a small autoencoder on an image or feature dataset; visualise the latent space and use the reconstruction error as an anomaly score on held-out corrupted samples.",
                "type": "code",
                "code": "# Problem 4 [Python]\nimport torch\nimport torch.nn as nn\nimport matplotlib.pyplot as plt\n\n# TODO: Train small autoencoder, visualise latent space, use reconstruction error for anomaly detection\n"
            },
            {
                "tag": "[Python]",
                "num": 5,
                "text": "Build a semantic / cross-modal search demo: embed a corpus with a pretrained sentence- or CLIP-style model, index it with FAISS, and measure recall@k and query latency against exact brute-force search as the index size grows.",
                "type": "code",
                "code": "# Problem 5 [Python]\nimport faiss\nimport torch\nfrom transformers import AutoTokenizer, AutoModel\n\n# TODO: Semantic / cross-modal search with FAISS vs exact brute-force\n"
            },
            {
                "tag": "[Analysis]",
                "num": 6,
                "text": "Write a one-page reflection tying the course together: pick one application area, name the main evaluation pitfalls (class imbalance, annotation quality, leakage) and one ethical concern (bias, privacy, or misuse) specific to unstructured data, and state how you would mitigate it.",
                "type": "markdown",
                "content": "### Problem 6 [Analysis]\n\n---\n**Course Synthesis & Ethical Reflection:**\n"
            }
        ]
    }
]

def make_cell(cell_type, text_or_lines):
    if isinstance(text_or_lines, str):
        lines = [line + "\n" for line in text_or_lines.split("\n")]
        # Remove trailing newline from last element if empty
        if lines and lines[-1] == "\n":
            lines[-1] = ""
    else:
        lines = text_or_lines
        
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

for ps in problem_sets:
    ps_dir = ps["id"]
    os.makedirs(ps_dir, exist_ok=True)
    nb_path = os.path.join(ps_dir, f"{ps['id']}.ipynb")
    
    cells = []
    
    # Title Header cell
    header_md = (
        f"# {ps['title']}\n\n"
        f"**Course:** Analysis of Unstructured Data  \n"
        f"**Instructor:** Janusz Szwabiński  \n"
        f"**Goal:** {ps['goal']}  \n\n"
        f"---"
    )
    cells.append(make_cell("markdown", header_md))
    
    # Setup / Imports code cell
    setup_code = (
        "# Setup environment & imports\n"
        "import sys\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n"
        "%config InlineBackend.figure_format = 'retina'\n"
    )
    cells.append(make_cell("code", setup_code))
    
    # Add cells for each problem
    for prob in ps["problems"]:
        if prob["type"] == "markdown":
            cells.append(make_cell("markdown", prob["content"]))
        elif prob["type"] == "code":
            prob_header = f"### Problem {prob['num']} {prob['tag']}\n\n**Task:**\n{prob['text']}\n"
            cells.append(make_cell("markdown", prob_header))
            cells.append(make_cell("code", prob["code"]))
    
    notebook = {
        "cells": cells,
        "metadata": NOTEBOOK_KERNEL_METADATA,
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated {nb_path}")
