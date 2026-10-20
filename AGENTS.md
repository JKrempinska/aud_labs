# AI Agent Guidelines & Repository Standards
**Course:** Analysis of Unstructured Data (AUD) — Masters Level  
**Instructor:** Janusz Szwabiński  
**Core Reference:** [AUD_Lab_assignments.pdf](file:///Users/juliakrempinska/Documents/PWR/masters/2_sem/aoud/lab/AUD_Lab_assignments.pdf)

---

## 1. Core Operating Principles

1. **Strict Fidelity to Assignment Requirements:**
   - Every problem set must be solved completely in its dedicated notebook (`problem_set_X/problem_set_X.ipynb`).
   - Maintain the tag structure: `[Theory]`, `[Python]`, and `[Analysis]`.
   - Do not deviate from or redefine problem statements; answer every explicit question and sub-clause.

2. **Language:**
   - All written text, mathematical derivations, analysis sections, code docstrings, and commit messages **must be in English**.

---

## 2. Mathematical Rigor & [Theory] Solutions

1. **No Shortcuts or Hand-waving:**
   - Show all algebraic steps explicitly. Do not jump from premise to conclusion with statements like "it is obvious that" or "by simple algebra".
2. **Structured Derivation Format:**
   Every `[Theory]` problem in Markdown must follow this structure:
   - **Assumptions & Notation:** Define variables, dimensions, norms, and conditions (e.g., $x, y \in \mathbb{R}^d$, $\|x\|_2 = 1$). State every assumption upfront.
   - **Formal Proposition:** Clear mathematical statement to be proved or derived.
   - **Step-by-Step Derivation:** Numbered or clearly separated mathematical steps with written justification for non-trivial transitions (e.g., "expanding the inner product", "applying Cauchy-Schwarz", "differentiating w.r.t. $z_k$").
   - **Sanity Checks & Edge Cases:** Check boundary values, dimensional consistency, or known special cases (e.g., orthogonal vectors, $\theta = 0$, $d=1$).
3. **LaTeX Formatting:**
   - Use clean LaTeX notation (e.g., `\mathbb{R}^d`, `\mathbf{x}`, `\Sigma`, `\operatorname{Var}`).
   - Use display math blocks (`$$ ... $$`) for key intermediate equations and final results.

---

## 3. Code Standards & [Python] Implementations

1. **Simplicity & Cleanliness:**
   - Write clear, idiomatic Python code.
   - **Avoid unnecessary comments:** Do not add trivial comments that simply restate what the syntax does (e.g., `# create list`, `# loop through items`, `# return x`).
   - Only use comments to explain non-obvious algorithmic choices or mathematical mappings.
2. **Reproducibility & Verification:**
   - Always set fixed random seeds (`np.random.seed(42)`, `torch.manual_seed(42)`) in cells involving stochastic operations.
   - **Enforce exact numerical tolerances:** When the assignment specifies a tolerance (e.g., *"agree to $10^{-6}$ on explained-variance ratios"*, *"gradient check to $< 10^{-5}$ relative error"*, *"max abs. error $< 10^{-5}$"*), **always include programmatic assertions** using:
     ```python
     np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
     ```
     or explicit `assert` statements that fail loudly if the condition is not met.
3. **Hardware & Model Selection:**
   - Prefer CPU/MPS-runnable lightweight models (e.g., `resnet18`, `distilbert-base-uncased`, or `sentence-transformers/all-MiniLM-L6-v2`).
   - Autodetect hardware acceleration:
     ```python
     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
     ```
4. **Data Management:**
   - Store all downloaded datasets or audio/image samples inside the gitignored `data/` directory.
   - Do not commit large binary data, model checkpoints, or temporary audio files to git.

---

## 4. Visualizations & [Analysis] Standards

1. **Publication-Ready Figures:**
   - Set readable figure sizes (e.g., `plt.figure(figsize=(8, 5))` or subplots).
   - Label all axes with descriptive titles and units where applicable.
   - Include distinct, accessible color palettes and clean legends.
   - Enable high-resolution rendering: `%config InlineBackend.figure_format = 'retina'`.
   - Call `plt.tight_layout()` to prevent label clipping.
2. **Lab-Report-Style Analysis:**
   - Keep analysis **detailed, technical, and concise** (no filler or marketing buzzwords).
   - Explicitly address each question raised in the prompt (e.g., why a metric fails, which embedding preserves global geometry, where over/under-fitting occurs, ethical considerations).
   - Tie empirical observations back to the mathematical theory established in earlier problems.

---

## 5. Execution & Validation Workflow

Before completing any task or problem:
1. **Execute the notebook cells** using the active virtual environment (`.venv`) and kernel `aoud-lab`.
2. Ensure the notebook executes cleanly **top-to-bottom without unhandled exceptions**.
3. Confirm that all figures render directly inside the notebook.

---

## 6. Git Hygiene & Commit Protocol

1. **Atomic Commits per Problem:**
   - Commit after solving each individual problem (or logical pair of derivation + code).
   - Do not batch an entire assignment into a single giant commit.
2. **Commit Message Convention:**
   - Use Conventional Commits with scope:
     - `feat(ps1): problem 1 - cosine similarity and euclidean ranking derivation`
     - `feat(ps1): problem 5 - feature vector loading and modality representation`
     - `feat(ps1): problem 6 - SVD PCA implementation and variance ratio verification`
     - `fix(ps2): adjust learning rate in MLP gradient check`
     - `docs(ps3): add bias probe analysis in problem 6`
3. **Repository Cleanliness:**
   - Keep git status clean.
   - Never commit `__pycache__`, `.ipynb_checkpoints`, virtual environments, model binaries (`*.pth`, `*.pt`, `*.bin`), or dataset directories.
