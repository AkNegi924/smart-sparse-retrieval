# Exp4Fuse Project — Final Report

**Project:** Reproduction of *Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion* (ACL 2025 Findings)  
**Repository / paper:** [arXiv:2506.04760](https://arxiv.org/abs/2506.04760)

---

## 1. Executive Summary

This project implements and runs the **Exp4Fuse** pipeline on **TREC Deep Learning 2019 (DL19) Passage** using the paper’s test cases, prompts, and evaluation setup. The pipeline uses **OpenRouter** with the free model **openai/gpt-oss-120b** for zero-shot hypothesis generation, **Pyserini** (BM25) for sparse retrieval, and **reciprocal rank fusion (RRF)** to combine the original-query and hypothesis-augmented runs. Evaluation is done with the **jtreceval** JAR for MAP, nDCG@10, and Recall@1000.

**Main result:** Exp4Fuse improves over BM25 on all three metrics (e.g. MAP 0.3013 → 0.3120, nDCG@10 0.5058 → 0.5289, Recall@1000 0.7501 → 0.7743), confirming the paper’s finding that fusing original and LLM-expanded runs helps sparse retrieval.

---

## 2. Concepts Used and Implemented

### 2.1 Exp4Fuse (Paper)

- **Two retrieval routes:**  
  - **Original route:** rank documents with the **original query** using a sparse retriever (BM25).  
  - **Query-expansion route:** use an LLM to generate a **hypothetical passage** for the query, build an **augmented query** (paper: \(q_e = \text{concat}(q_o \times \lambda,\, r_q)\) with \(\lambda=5\)), then rank with the **same** sparse retriever.
- **Fusion:** The two ranked lists are merged with a **modified reciprocal rank fusion (RRF)** so that documents that appear in both lists get a boost. Final ranking is by the fused score.
- **Indirect use of the LLM:** The LLM is not used to score documents; it only expands the query. The actual scoring is done by the sparse retriever on both the original and the expanded query.

### 2.2 Implemented Components

| Concept | Implementation |
|--------|-----------------|
| **Zero-shot hypothesis generation** | One LLM call per topic: system prompt *“Please write a passage to answer the question.”* (paper’s TREC DL19 prompt), user = query. Model: OpenRouter `openai/gpt-oss-120b:free`. Max 128 tokens, temperature 0.6, top_p 0.9 (paper §4.1). |
| **Augmented query** | \(q_e = (q_o + \text{'.'}) \times 5 + r_q\): original query (with trailing period) repeated 5 times, then concatenated with the hypothetical passage (paper \(\lambda=5\)). |
| **Sparse retrieval** | **BM25** via Pyserini’s `LuceneSearcher` on the prebuilt index **msmarco-v1-passage**. Top-1000 documents per query. |
| **Modified RRF** | Paper formula: \(FR_{\text{score}} = (w_i + n/10) \cdot \sum_i 1/(k + r_i)\) with \(k=60\), \(w_1=w_2=1\). \(n\) = number of lists containing the document (1 or 2). Implemented as in the paper/notebook. |
| **Evaluation** | **jtreceval** JAR (Pyserini’s trec_eval). Metrics: **MAP**, **nDCG@10**, **Recall@1000** on TREC DL19 passage qrels (`dl19-passage`). |

---

## 3. Full Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                      │
│  • TREC DL19 data/dl19_topics  (43 topics: [qid, query])                    │
│  • (Optional) TREC DL19 data/dl19_hypothesis  or  generate via LLM          │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
┌───────────────────────────────┐         ┌───────────────────────────────────┐
│  STEP 1 (optional)            │         │  STEP 2: Retrieval + Fusion        │
│  generate_dl19_hypothesis.py  │         │  run_exp4fuse_dl19.py              │
│  • Read dl19_topics           │         │  • Copy/create dl19_topics,        │
│  • For each query:            │         │    dl19_hy in Exp4Fuse/            │
│    - Call OpenRouter          │         │  • Load BM25 index                 │
│      (gpt-oss-120b:free)      │         │    msmarco-v1-passage             │
│    - System: "Please write    │         │  • Run 1: original query →         │
│      a passage to answer      │         │    dl19_bm25                       │
│      the question."           │         │  • Run 2: (q×5 + hypothesis) →     │
│    - User: query              │         │    dl19_bm25_hy                    │
│  • Write dl19_hypothesis,     │         │  • Fuse (RRF k=60) →                │
│    dl19_hy                    │         │    dl19_bm25_Exp4Fuse              │
└───────────────────────────────┘         │  • Evaluate both runs with         │
                    │                     │    jtreceval (MAP, nDCG@10, R@1k)   │
                    └────────────────────┼───────────────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                     │
│  • Run files: dl19_bm25, dl19_bm25_hy, dl19_bm25_Exp4Fuse (TREC format)     │
│  • Console: BM25 vs BM25+Exp4Fuse metrics                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Step-by-Step

1. **Hypothesis generation (optional)**  
   - **Script:** `generate_dl19_hypothesis.py`  
   - **Input:** `TREC DL19 data/dl19_topics` (JSON list of `[qid, query]`).  
   - **Process:** For each of the 43 topics, one OpenRouter chat call with the paper’s TREC DL19 prompt; response truncated to 128 tokens.  
   - **Output:** `TREC DL19 data/dl19_hypothesis` and `dl19_hy` (JSON list of `[qid, query, hypothesis_text]`).  
   - **Config:** `.env` with `OPENROUTER_API_KEY` (in Exp4Fuse or parent).

2. **Retrieval and fusion**  
   - **Script:** `run_exp4fuse_dl19.py`  
   - **Input:** `dl19_topics` and `dl19_hy` (created or copied from `TREC DL19 data/` if missing).  
   - **Process:**  
     - Load Pyserini BM25 index `msmarco-v1-passage`.  
     - **Original run:** For each topic, search with the original query → write `dl19_bm25`.  
     - **Hypothesis run:** For each hypothesis row, build \((query + '.')\times 5 + hypothesis\), search → write `dl19_bm25_hy`.  
     - **Fusion:** Merge the two runs with modified RRF (k=60) → write `dl19_bm25_Exp4Fuse`.  
   - **Evaluation:** Run jtreceval JAR on `dl19-passage` qrels for MAP, nDCG@10, Recall@1000 on `dl19_bm25` and `dl19_bm25_Exp4Fuse`; print results.

3. **Evaluation**  
   - **Tool:** jtreceval JAR (downloaded via Pyserini’s `download_evaluation_script("trec_eval")`).  
   - **Qrels:** Pyserini’s `get_qrels_file("dl19-passage")` (TREC DL19 passage relevance judgments).  
   - **Metrics:** MAP, nDCG@10, Recall@1000 (paper’s TREC DL19 metrics).

---

## 4. What Was Implemented (Code)

### 4.1 Files and Roles

| File | Purpose |
|------|--------|
| `generate_dl19_hypothesis.py` | Reads `TREC DL19 data/dl19_topics`, calls OpenRouter (gpt-oss-120b:free) with the paper’s TREC DL19 prompt, writes `TREC DL19 data/dl19_hypothesis` and `dl19_hy`. |
| `run_exp4fuse_dl19.py` | Ensures `dl19_topics` and `dl19_hy` exist (copy from TREC DL19 data if needed); runs BM25 original + BM25 hypothesis-augmented; fuses with RRF k=60; runs jtreceval and prints MAP, nDCG@10, Recall@1000. |
| `TREC DL19 data/dl19_topics` | Input: 43 TREC DL19 topics as JSON `[[qid, query], ...]`. |
| `TREC DL19 data/dl19_hypothesis` | Input/Output: 43 entries as JSON `[[qid, query, hypothesis], ...]`. Either pre-existing (e.g. from the repo) or produced by `generate_dl19_hypothesis.py`. |
| `README.md` | Setup, requirements (JDK 21, pyserini), and run instructions. |
| `install.sh` | Paper’s dependency install script. |

### 4.2 Dependencies

- **Python:** `openai` (for OpenRouter), `pyserini` (retrieval + qrels).  
- **Java:** JDK 21 (required for Pyserini’s Anserini JARs).  
- **Data/indices:** Pyserini prebuilt index `msmarco-v1-passage` (downloaded on first run); jtreceval JAR and DL19 passage qrels obtained via Pyserini helpers.

---

## 5. Output Files Generated

### 5.1 Intermediate and Run Files (in `Exp4Fuse/`)

| File | Format | Description |
|------|--------|-------------|
| `dl19_topics` | JSON | Copy of `TREC DL19 data/dl19_topics` (43 topics). Created by run script if missing. |
| `dl19_hy` | JSON | Copy of hypothesis data (43 rows `[qid, query, hypothesis]`). Created by run script if missing. |
| `dl19_bm25` | TREC run | BM25 ranking with **original query** only. One line per hit: `qid Q0 docid rank score runid`. |
| `dl19_bm25_hy` | TREC run | BM25 ranking with **augmented query** (query×5 + hypothesis). Same format. |
| `dl19_bm25_Exp4Fuse` | TREC run | **Fused** ranking (modified RRF of the two runs). Same format; used for reported Exp4Fuse metrics. |

### 5.2 TREC Run File Format

Each line:

```
qid Q0 docid rank score runid
```

Example:

```
264014 Q0 5611210 1 15.78 rank
264014 Q0 6641238 2 15.09 rank
...
```

- **qid:** TREC topic id.  
- **Q0:** literal.  
- **docid:** MS MARCO passage id.  
- **rank:** 1–1000 per topic.  
- **score:** BM25 score (original/hy) or RRF score (Exp4Fuse).  
- **runid:** `rank`.

---

## 6. Detailed Output and Analysis

### 6.1 Console Output (Example Run)

Typical output of `python run_exp4fuse_dl19.py`:

```
Loading BM25 index (msmarco-v1-passage)...
BM25 original query run -> dl19_bm25
BM25 hypothesis-augmented run -> dl19_bm25_hy
Fusion (RRF k=60) -> dl19_bm25_Exp4Fuse

--- TREC DL19 evaluation (paper metrics) ---
BM25 (original):
  map: 0.3013
  ndcg_cut_10: 0.5058
  recall_1000: 0.7501
BM25 + Exp4Fuse:
  map: 0.3120
  ndcg_cut_10: 0.5289
  recall_1000: 0.7743

Done.
```

(The first time, the jtreceval JAR is downloaded to `~/.cache/pyserini/eval/`; subsequent runs skip the download.)

### 6.2 Numerical Results

| System | MAP | nDCG@10 | Recall@1000 |
|--------|-----|---------|-------------|
| **BM25 (original)** | 0.3013 | 0.5058 | 0.7501 |
| **BM25 + Exp4Fuse** | 0.3120 | 0.5289 | 0.7743 |
| **Δ (Exp4Fuse − BM25)** | **+0.0107** | **+0.0231** | **+0.0242** |

### 6.3 Analysis

- **Baseline:** BM25-only scores match the expected TREC DL19 passage range (e.g. paper Table 1 reports BM25 MAP ~0.301, nDCG@10 ~0.506).  
- **Exp4Fuse gain:** Fusing the original-query run with the hypothesis-augmented run improves all three metrics. The relative gains are about **+3.6%** MAP, **+4.6%** nDCG@10, and **+3.2%** Recall@1000.  
- **Mechanism:** The hypothetical passages add relevant terms and paraphrases; the second run retrieves some relevant documents that BM25 on the original query misses. RRF keeps strong original-query results while promoting documents that appear in both runs, so the fused list is better than either run alone.  
- **Reproducibility:** This run uses OpenRouter’s gpt-oss-120b for hypotheses; the paper used gpt-4o-mini. Slight differences in metrics across runs are expected; the trend (Exp4Fuse > BM25) is consistent with the paper.

---

## 7. How to Reproduce

### 7.1 Environment

- Python 3 with `openai` and `pyserini`.  
- **JDK 21** (e.g. `openjdk-21-jdk`), set as default `java`/`javac`.  
- `.env` in Exp4Fuse (or parent) with `OPENROUTER_API_KEY` if you (re)generate hypotheses.

### 7.2 Commands

```bash
cd Exp4Fuse

# Optional: regenerate hypotheses with OpenRouter (openai/gpt-oss-120b:free)
python generate_dl19_hypothesis.py

# Run retrieval, fusion, and evaluation
python run_exp4fuse_dl19.py
```

If you skip hypothesis generation, the script uses existing `TREC DL19 data/dl19_hypothesis` (and copies to `dl19_hy` as needed).

---

## 8. References

- **Paper:** Liu & Zhang (2025), *Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion*, ACL 2025 Findings. [arXiv:2506.04760](https://arxiv.org/abs/2506.04760)  
- **Pyserini:** [github.com/castorini/pyserini](https://github.com/castorini/pyserini)  
- **TREC DL 2019:** [TREC 2019 Deep Learning Track](https://microsoft.github.io/TREC-2019-Deep-Learning/)  
- **OpenRouter:** [openrouter.ai](https://openrouter.ai) (API used for gpt-oss-120b:free).

---

*Report generated for the Exp4Fuse reproduction project.*
