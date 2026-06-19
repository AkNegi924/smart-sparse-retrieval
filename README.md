# Smart Sparse Retrieval: Exp4Fuse Reproduction & Extensions

This repository contains the reproduction and research extensions of **Exp4Fuse**, a query-expansion and rank-fusion framework for sparse retrieval systems, accepted at the Findings of **ACL 2025**.

The project is structured as an Information Retrieval course project focusing on **Budget-Constrained Sparse Retrieval**. We systematically analyze and extend the baseline Exp4Fuse framework under strict practical limits (no additional LLM API costs or new datasets), achieving substantial improvements in retrieval effectiveness.

---

## 📌 Project Overview & Deliverables
- 📄 **Project Report**: [Report.pdf](file:///d:/ActiveProjects/smart-sparse-retrieval/Report.pdf) (Detailed methodologies, findings, and analysis)
- 📊 **Presentation Slides**: [presentation.pdf](file:///d:/ActiveProjects/smart-sparse-retrieval/presentation.pdf) (Summary deck for evaluation/viva)
- 💻 **Source Code**: [code/](file:///d:/ActiveProjects/smart-sparse-retrieval/code) (Modular Python implementation, datasets, runs, and results)

---

## 🔍 Core Concept: Exp4Fuse

Exp4Fuse improves sparse retrieval (e.g., BM25) by utilizing Large Language Models (LLMs) for zero-shot query expansion and fusing the results with the original query's retrieval runs.

```
                  ┌───────────────────┐
                  │  Original Query   │
                  └─────────┬─────────┘
            ┌───────────────┴───────────────┐
            ▼                               ▼
     [Original Route]             [LLM-Expansion Route]
   Retrieve via BM25 on            Generate Hypothesis
     original query                  Passage via LLM
            │                               │
            │                     Build Augmented Query:
            │              q_e = (q_o + '.') * λ + Hypothesis
            │                               │
            │                     Retrieve via BM25 on
            │                       augmented query
            ▼                               ▼
      Ranked Run 1                    Ranked Run 2
            └───────────────┬───────────────┘
                            ▼
                    [Modified RRF Fusion]
                  Boost items in both lists
                            ▼
                    Final Ranked Results
```

### 1. Two-Route Retrieval
* **Original Route**: Query $q_o$ retrieves documents using BM25.
* **LLM-Expansion Route**: Query $q_o$ is sent to an LLM to generate a zero-shot hypothetical response passage $r_q$. An augmented query $q_e$ is built by concatenating the original query repeated $\lambda$ times with the hypothetical passage:
  \[ q_e = \text{concat}(q_o \times \lambda, r_q) \]
  A second retrieval run is executed using $q_e$.

### 2. Reciprocal Rank Fusion (RRF)
The two runs are fused using a modified RRF formula where documents appearing in both lists receive a score boost based on the number of overlapping lists $n$:
\[ \text{Score}(d) = \left(w_i + \frac{n}{10}\right) \cdot \sum_{i \in \text{Runs}} \frac{1}{k + r_i(d)} \]

---

## 🚀 Implemented Extensions (Novelties)

We refactored the baseline code into a modular framework ([retrieval.py](file:///d:/ActiveProjects/smart-sparse-retrieval/code/retrieval.py), [fusion.py](file:///d:/ActiveProjects/smart-sparse-retrieval/code/fusion.py), [eval_util.py](file:///d:/ActiveProjects/smart-sparse-retrieval/code/eval_util.py)) and implemented three novelty tracks:

### 1. Hyperparameter Sensitivity & Grid Tuning (Novelty 1)
* **Goal**: Optimize the expansion repetition factor $\lambda$, the RRF constant $k$, and the route weights $w_1$ and $w_2$.
* **Outcome**: Discovered that the paper's default setup ($\lambda=5, k=60$) is suboptimal. Tuning to **$\lambda=4, k=30, w_1=1.0, w_2=1.0$** yielded significant gains across all metrics.

### 2. Fusion Method Ablation (Novelty 2)
* **Goal**: Compare rank-only RRF with score-aware and simpler fusion alternatives.
* **Methods Evaluated**: Modified RRF, Standard RRF, Score-Linear Fusion, and CombSUM.
* **Outcome**: **Score-Linear Fusion ($\alpha=0.5$)** and **CombSUM** achieved the highest top-rank precision (**nDCG@10 of 0.5626**), outperforming rank-only RRF by +3.7%.

### 3. Hypothesis Truncation (Novelty 3)
* **Goal**: Evaluate if truncating hypothetical passages to a prefix (first 64 or 96 tokens) reduces term-frequency noise.
* **Outcome**: Truncating did not affect performance in this dataset, indicating that the full generated hypothesis is robustly handled or that BM25 handles longer queries with the same efficiency in this setup.

---

## 📊 Experimental Results

Experiments conducted on the **TREC Deep Learning 2019 (DL19) Passage** dataset (43 query topics) using a BM25 retriever on the prebuilt `msmarco-v1-passage` index.

| System | MAP | nDCG@10 | Recall@1000 |
| :--- | :---: | :---: | :---: |
| **BM25 (Original)** | 0.3013 | 0.5058 | 0.7501 |
| **Exp4Fuse Baseline** | 0.3208 | 0.5424 | 0.7756 |
| **Best Novelty 1 (Tuned Grid)** | **0.3330** *(+3.8%)* | 0.5519 *(+1.8%)* | **0.7855** *(+1.3%)* |
| **Best Novelty 2 (Score-Linear/CombSUM)** | 0.3309 *(+3.1%)* | **0.5626** *(+3.7%)* | 0.7765 *(+0.1%)* |

*Note: Percentage gains shown relative to the Exp4Fuse Baseline.*

---

## 📂 Repository Directory Structure

```text
smart-sparse-retrieval/
├── Report.pdf                      # Course project final report (PDF)
├── presentation.pdf                # Project presentation slides (PDF)
└── code/                           # Implementation files
    ├── generate_dl19_hypothesis.py # LLM zero-shot query expansion script
    ├── run_exp4fuse_dl19.py        # Baseline paper reproduction script
    ├── run_experiments.py          # Orchestrator for Novelty 1-3 experiments
    ├── retrieval.py                # Retrieval logic (BM25 wrapper)
    ├── fusion.py                   # RRF, Standard RRF, CombSUM & Linear Fusion
    ├── eval_util.py                # Wrapper for trec_eval metrics calculations
    ├── requirements.txt            # Python dependencies
    ├── install.sh                  # Setup script for baseline dependencies
    ├── dl19_topics                 # TREC DL19 topics JSON data
    ├── dl19_hy                     # Generated hypotheses JSON data
    ├── results/                    # CSVs of results from run_experiments.py
    └── runs/                       # TREC-format retrieval run output files
```

---

## ⚙️ Setup & Installation

### Prerequisites
1. **Python 3.10+** (Tested on Python 3.12)
2. **Java 21 JDK** (Required by Pyserini for Java Lucene indices)

### Quick Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AkNegi924/smart-sparse-retrieval.git
   cd smart-sparse-retrieval/code
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   # On Linux/macOS
   source .venv/bin/activate
   # On Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   *(Ensure you have Java 21 configured. Check your system's Java version with `java -version`)*.

---

## 🏃 Running Experiments

All reproduction run commands should be executed from the `code/` directory with the virtual environment active.

### 1. Reproducing Baseline Parity
Verify the basic Exp4Fuse implementation against BM25:
```bash
python run_experiments.py --novelty baseline
```

### 2. Running Novelty 1 (Hyperparameter Tuning Grid)
Run the parameter grid sweep for $\lambda \in \{4,5,6\}$, $k \in \{30,60\}$, and weights:
```bash
python run_experiments.py --novelty 1
```

### 3. Running Novelty 2 (Fusion Method Ablation)
Compare Modified RRF, Standard RRF, Score-Linear, and CombSUM:
```bash
python run_experiments.py --novelty 2
```

### 4. Running Novelty 3 (Truncation Ablation)
Evaluate the impact of truncating the expansion text:
```bash
python run_experiments.py --novelty 3
```

### 5. Run All Experiments Simultaneously
Run baseline validation and all novelties in a single sweep:
```bash
python run_experiments.py --novelty all
```
*Outputs will be saved in `results/` (metrics summary CSVs) and `runs/` (TREC search result outputs).*

---

## 📄 References & Citation

### Paper Citation
```bibtex
@misc{liu2025exp4fuserankfusionframework,
      title={Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion}, 
      author={Lingyuan Liu and Mengxiang Zhang},
      year={2025},
      eprint={2506.04760},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2506.04760}, 
}
```
