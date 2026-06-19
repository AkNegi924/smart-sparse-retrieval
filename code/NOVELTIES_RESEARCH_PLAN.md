# Exp4Fuse Extensions: Detailed Research Plan and Novelty Specification

**Base paper:** *Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion* (ACL 2025 Findings)  
**Scope:** Research-grade extensions on TREC Deep Learning 2019 (DL19) Passage, without additional LLM API cost or new datasets.  
**Target metrics:** MAP, nDCG@10, Recall@1000.

---

## Document purpose

This document specifies the **top five novelties** planned for a research paper built on Exp4Fuse. For each novelty we state: **why** it is needed, **how** it will be implemented (approach in detail), **how** it is expected to change outcomes, **importance** for the contribution, **limitations** in the current (paper’s) implementation and in our own setup, and **research limitations** we acknowledge. Where the base paper already recommends or uses a specific choice, we note it and clarify the scope for novelty.

---

## Overview of the five novelties

| # | Novelty | One-line summary |
|---|--------|-------------------|
| 1 | **Hyperparameter tuning and sensitivity analysis** | Systematic study of λ, k, and route weights (w₁, w₂); optional simple adaptive rule. |
| 2 | **Fusion method ablation** | Compare modified RRF (paper), standard RRF, score-based fusion, and CombSUM/Borda on the same runs. |
| 3 | **Hypothesis truncation and quality-aware use** | Use only the first N tokens (or sentences) of the hypothesis for expansion; analyze when it helps. |
| 4 | **Three-way fusion from a single hypothesis** | Add a third retrieval route (e.g. half-hypothesis or different λ) and fuse with 3-way RRF/CombSUM; compare 2-way vs 3-way. |
| 5 | **Deeper retrieval and score-based re-ranking** | Retrieve more than 1000 per route and/or re-rank the fused set by original-query BM25 score. |

---

## Research and scope constraints (acknowledged limitations)

- **No additional LLM calls:** All extensions use the same per-query hypothesis already generated; no new API cost.
- **Single dataset:** Experiments are on TREC DL19 Passage only; generalisation to other datasets is not evaluated and is a stated limitation.
- **No caching, batching, or latency optimisations:** We do not introduce caching of expansions or streaming fusion; the focus is on retrieval and fusion effectiveness.
- **Sparse retriever:** We use BM25 (Pyserini); learned sparse retrievers (e.g. SPLADE) are not in scope for this plan but can be mentioned as future work.
- **Heuristic adaptation only:** Any “adaptive” behaviour (e.g. λ or weights by query/hypothesis length) uses hand-crafted rules or simple statistics, not learned models.

---

# Novelty 1: Hyperparameter tuning and sensitivity analysis

## 1.1 Why this novelty

The base paper fixes **λ = 5** (query repetition in the augmented query), **k = 60** (RRF constant), and **w₁ = w₂ = 1** (route weights) for all queries and datasets. It states that k was chosen in a pilot study but does not report sensitivity to λ or to route weights, and does not vary them by query or dataset. In information retrieval, fusion and expansion hyperparameters are known to be dataset- and sometimes query-dependent; a single fixed configuration may be suboptimal. This novelty addresses that gap by (1) systematically varying these parameters and (2) optionally deriving a simple, interpretable rule (e.g. based on query or hypothesis length) for choosing or adjusting them.

## 1.2 What the base paper already does

The paper uses **λ = 5** to balance the influence of the original query and the hypothetical document in the augmented query (Equation 1). It uses **k = 60** in the modified RRF formula (Equation 2) and states this was fixed during a pilot study. It sets **w₁ = w₂ = 1** so both routes contribute equally. The paper does not claim these are optimal; it simply reports results with these settings. Thus there is clear scope for novelty: we do not contradict the paper but extend it by providing the first systematic sensitivity analysis and, if beneficial, a simple adaptive rule.

## 1.3 Approach in detail

- **Parameter grid (no adaptation):**
  - **λ:** Try values in {3, 4, 5, 6, 7}. For each λ, form the augmented query as \(\text{concat}(q_o \times \lambda,\, r_q)\) (or the same with a trailing period as in the current implementation). Keep all other settings (k, w₁, w₂) fixed at the paper’s values. Run retrieval and fusion for each λ; evaluate MAP, nDCG@10, Recall@1000.
  - **k:** Try k ∈ {30, 60, 100} with λ = 5 and w₁ = w₂ = 1. For each k, run the same two retrieval runs and apply the paper’s modified RRF; report the same metrics.
  - **Route weights (w₁, w₂):** Try (1, 1), (1.2, 0.8), (0.8, 1.2), and optionally (1.5, 0.5) and (0.5, 1.5) with λ = 5 and k = 60. The fusion formula remains the paper’s modified RRF, with w₁ and w₂ applied as in the paper (e.g. if the formula is written per-route, each route uses its corresponding weight).
- **Sensitivity analysis:** For each parameter, report a small table or figure: parameter value vs MAP, nDCG@10, R@1000. Comment on whether the optimum is at the paper’s default or elsewhere, and whether the curve is flat or sensitive.
- **Optional adaptive rule:** After the grid, define a simple rule using only available signals (no extra LLM):
  - **Query length:** e.g. if query word count &lt; 5, use λ = 6 or 7; if ≥ 10, use λ = 3 or 4; else λ = 5. Rationale: short queries may benefit from more repetition to balance a long hypothesis.
  - **Hypothesis length:** e.g. if hypothesis token count &lt; 20, use (w₁, w₂) = (1.2, 0.8) to down-weight the expansion route. Rationale: very short hypotheses may be low quality.
  Implement this rule for each topic, run fusion with the selected (λ, k, w₁, w₂), and compare to the best fixed configuration and to the paper’s default.

## 1.4 How it will make a difference

- **Fixed tuning:** Finding a better (λ, k, w₁, w₂) than the paper’s default would directly improve MAP, nDCG@10, and/or R@1000 on DL19, and would provide a concrete recommended setting for practitioners.
- **Adaptive rule:** If the rule-based adaptation beats the best fixed config (or the paper’s default), it would show that query/hypothesis-dependent hyperparameters can improve over one-size-fits-all, without any extra LLM cost. If it does not, the analysis still adds value by showing that fixed tuning is sufficient on this dataset.

## 1.5 Importance for the research contribution

- Fills a clear gap: the paper does not tune or analyse these parameters.
- Low risk: no change to the number of retrievals or LLM calls; only configuration and optional per-query rules.
- Results are directly usable: best settings and (if applicable) the adaptive rule can be reported and reused.
- Strengthens the paper’s reproducibility and practical usefulness.

## 1.6 Limitations in current (paper’s) implementation

- No sensitivity analysis or ablation over λ, k, or w₁, w₂.
- No query- or dataset-dependent adaptation of these parameters.
- No justification beyond a pilot for k = 60.

## 1.7 Research limitations (our work)

- Only one dataset (DL19); optimal or adaptive settings may not generalise.
- Adaptive rule is heuristic; no learning or validation on a separate set.
- We do not use a separate validation set to pick the best config; we report performance on the same test set used for comparison (we can state this explicitly and treat it as development-set tuning for reporting).

---

# Novelty 2: Fusion method ablation

## 2.1 Why this novelty

Exp4Fuse uses a **modified** reciprocal rank fusion formula that adds a term involving the number of lists n in which a document appears (e.g. \(w_i + n/10\)) and uses fixed weights w₁, w₂. The paper does not compare this to other well-known fusion methods (standard RRF, CombSUM, Borda count, or score-based linear combination). In the broader IR literature, the choice of fusion method can significantly affect effectiveness; the same two ranked lists can yield different final rankings under different fusion schemes. Without a comparison, it is unclear whether the modified RRF is the best choice or whether simpler or more interpretable methods would perform as well or better. This novelty provides the first systematic comparison of fusion methods in the Exp4Fuse setting.

## 2.2 What the base paper already does

The paper motivates RRF by the idea that lower-ranked documents still matter (unlike in some exponential schemes) and introduces an “adaptive weight strategy” so that documents appearing in both lists get a boost. The exact formula is \(FR_{\text{score}} = (w_i + n/10) \cdot \sum_i 1/(k + r_i)\) with k = 60 and w₁ = w₂ = 1. The paper does not ablate against other fusion methods or justify this specific form beyond intuition. So the scope for novelty is to keep the same two retrieval runs and compare multiple fusion methods on the same inputs.

## 2.3 Approach in detail

- **Runs:** Use a single set of two runs for all fusion experiments: (1) original-query BM25 run, (2) hypothesis-augmented BM25 run (paper’s λ = 5, or best λ from Novelty 1 if available). No additional retrieval.

- **Fusion methods to implement:**
  1. **Paper’s modified RRF (baseline):** \(FR_{\text{score}}(d) = (w_i + n/10) \cdot \sum_{i=1}^{2} 1/(k + r_i(d))\), with n = number of lists containing d, k = 60, w₁ = w₂ = 1.
  2. **Standard RRF:** \(s(d) = \sum_{i=1}^{2} 1/(k + r_i(d))\), same k = 60, no n/10 term and no per-route weights. This is the classic RRF used in many fusion studies.
  3. **Score-based fusion (linear combination):** For each document d that appears in at least one run, obtain BM25 scores from both runs (if d is missing from a run, assign a score below the minimum observed, or use 0). Normalise scores per topic (e.g. min-max to [0,1] or z-score). Then \(s(d) = \alpha \cdot \tilde{s}_1(d) + (1-\alpha) \cdot \tilde{s}_2(d)\). Try α ∈ {0.3, 0.5, 0.7}. Rank by s(d).
  4. **CombSUM:** Same normalisation as above; \(s(d) = \tilde{s}_1(d) + \tilde{s}_2(d)\) (or weighted sum with a single weight). No rank-based component.
  5. **Borda count (optional):** For each run, assign a score that is a function of rank (e.g. (K − rank + 1) for top-K). Sum the scores across the two runs; rank by total. This is a pure rank-based method.

- **Evaluation:** For each method (and each α for score-based), produce the final ranking and evaluate MAP, nDCG@10, Recall@1000. Report a table: method (and α if applicable) vs the three metrics. Identify the best-performing method and compare it to the paper’s modified RRF.

- **Discussion:** Comment on whether the paper’s modified RRF is best, and if not, what the best method is and why it might work better (e.g. score-based fusion using magnitude of relevance scores vs rank-only RRF).

## 2.4 How it will make a difference

- If another method (e.g. score-based or standard RRF) beats the paper’s modified RRF, we improve metrics and provide an alternative that may be simpler or more interpretable.
- If the paper’s method remains best, we still add value by showing that the proposed modification is justified compared to standard alternatives.
- In either case, the ablation is a necessary step for a research paper: it answers “why this fusion method?”

## 2.5 Importance for the research contribution

- Addresses a direct limitation: the paper does not compare with other fusion methods.
- No extra cost: same two runs, multiple fusion post-processing steps.
- Standard practice in IR: fusion ablations are expected in rank-fusion work.
- Provides a clear recommendation for practitioners (which fusion to use with Exp4Fuse-style runs).

## 2.6 Limitations in current (paper’s) implementation

- No comparison with standard RRF, CombSUM, Borda, or score-based fusion.
- No justification for the specific form of the modified RRF (n/10 term, role of w_i).
- No ablation on the fusion component.

## 2.7 Research limitations (our work)

- Evaluation on one dataset only; best method may be dataset-dependent.
- Score-based fusion requires handling documents that appear in only one run (imputation or exclusion); we will specify the choice and keep it consistent.
- We do not learn fusion weights or use a learned fusion model; all methods are fixed formulas.

---

# Novelty 3: Hypothesis truncation and quality-aware use

## 3.1 Why this novelty

The hypothetical document \(r_q\) is generated with a maximum length (e.g. 128 tokens in the paper). Longer hypotheses can introduce irrelevant or redundant content toward the end, which may dilute the effect of the original query and of the most relevant part of the hypothesis in sparse retrieval (BM25 is sensitive to term frequency and length). The paper uses the full hypothesis without any truncation or quality-based filtering. Using only the first portion of the hypothesis (e.g. first 64 or 96 tokens, or first 2–3 sentences) could reduce noise while retaining the most query-focused content, which often appears at the beginning of the generated passage. This novelty tests that hypothesis and, if beneficial, analyses when truncation helps (e.g. for long vs short hypotheses).

## 3.2 What the base paper already does

The paper concatenates the full hypothetical document with the repeated original query (Equation 1). It does not truncate or filter the hypothesis, and does not analyse hypothesis length or quality. So there is scope for novelty in how we use the **same** generated hypothesis (no extra LLM call) by restricting the expansion to a prefix.

## 3.3 Approach in detail

- **Truncation variants:** From the existing hypothesis text (after any existing sanitisation), form augmented queries as follows:
  - **Full (baseline):** \(q_e = \text{concat}(q_o \times \lambda,\, r_q)\) as in the paper.
  - **Token-based truncation:** Use only the first 32, 64, or 96 tokens of \(r_q\) (tokenisation can be simple whitespace-based or use a lightweight tokeniser). \(q_e = \text{concat}(q_o \times \lambda,\, r_q^{:N})\).
  - **Sentence-based truncation (optional):** Use only the first 1 or 2 sentences of \(r_q\) (sentence split by period + space or a simple heuristic). Compare to full and to one token-based cutoff.
- **Retrieval:** For each truncation variant, produce one hypothesis-augmented run (same original-query run for all). Fuse each augmented run with the original run using the same fusion method (e.g. paper’s modified RRF or the best from Novelty 2). Evaluate MAP, nDCG@10, Recall@1000.
- **Analysis:** Report metrics by truncation length. Optionally, stratify by hypothesis length (e.g. short vs long): for each topic, classify hypothesis as “short” or “long” (e.g. by median length), and report whether truncation helps more for long hypotheses. This addresses “when does truncation help?”

## 3.4 How it will make a difference

- If a truncated variant (e.g. first 64 tokens) beats full hypothesis, we improve metrics and give a simple, interpretable recipe: “use the first N tokens of the hypothesis.”
- If full hypothesis remains best, we still add evidence that the full expansion is useful on this dataset and that aggressive truncation can hurt.
- Stratified analysis (by hypothesis length) can show that truncation is especially useful when hypotheses are long, supporting a quality-aware use of the expansion.

## 3.5 Importance for the research contribution

- Addresses an unexplored dimension: the paper does not consider hypothesis length or quality when building the augmented query.
- Zero extra cost: same hypotheses, different substring used.
- Simple to implement and explain; good for reproducibility.
- Can inform future work on hypothesis quality estimation (e.g. learned truncation or filtering).

## 3.6 Limitations in current (paper’s) implementation

- No truncation or filtering of the hypothesis.
- No analysis of hypothesis length or quality.
- No handling of very short or clearly low-quality hypotheses (e.g. single sentence, repetition of the query).

## 3.7 Research limitations (our work)

- Truncation lengths (32, 64, 96) are chosen heuristically; we do not learn an optimal cutoff.
- Tokenisation may be approximate (e.g. whitespace) if no proper tokeniser is used; we will state the choice.
- “Quality” is only approximated by length and truncation; we do not use a separate quality model or NLI.

---

# Novelty 4: Three-way fusion from a single hypothesis

## 4.1 Why this novelty

The base paper’s ablation (Figure 2 and Table 4) shows that adding a **third** retrieval route (e.g. original + hypothetical-document expansion + multiple-query expansion) can improve over two routes on TREC DL19/20, but adding a fourth route can hurt. The paper concludes that the two-route setup (original + hypothetical-document) is a good cost–performance trade-off, especially because additional routes in their setup required additional LLM-generated content. In our setting, we do **not** want extra LLM calls. We can, however, create a **third route from the same hypothesis** by changing only how we form the augmented query (e.g. using the first half of the hypothesis, or a different λ). That gives a third ranked list without any additional API cost, and we can then fuse the three lists with a 3-way extension of the fusion method. This tests whether the paper’s finding “3 routes can help” carries over when the third route is a different view of the same hypothesis, and whether 3-way fusion improves over 2-way on DL19.

## 4.2 What the base paper already does

The paper uses two routes: original query and hypothetical-document expansion (with λ = 5). It experiments with three and four routes by adding “multiple query expansion” and “step-back query expansion,” which require additional LLM-generated content. It does not explore a third route that reuses the same hypothesis (e.g. half-hypothesis or different λ). So the scope for novelty is: (1) define one or two third-route variants that use the same \(r_q\), (2) extend the fusion formula to three lists, and (3) compare 2-way vs 3-way and optionally compare different ways to form the third route.

## 4.3 Approach in detail

- **Third-route variants (choose one or both):**
  - **Half-hypothesis route:** Augmented query = \(q_o \times \lambda\) concatenated with the **first half** of \(r_q\) (by character count or by token count). Same λ as the main route (e.g. 5). This yields a run that emphasises the beginning of the hypothesis.
  - **Different-λ route:** Augmented query = \(q_o \times \lambda'\) concatenated with full \(r_q\), with \(\lambda' \neq \lambda\) (e.g. λ = 5 for the main expansion route and λ′ = 3 for the third route). This yields a run that gives relatively more or less weight to the original query in the expansion.
- **Retrieval:** Produce three runs: (1) original query only, (2) \(q_o \times \lambda + r_q\) (full hypothesis), (3) either half-hypothesis or different-λ (or both, giving three or four runs in total; for a clean 3-way comparison we have exactly three runs). All with the same retriever and same k=1000 (or same depth as in Novelty 5 if applicable).

- **3-way fusion formula:** Extend the paper’s modified RRF to three lists. For each document d, let \(r_1(d), r_2(d), r_3(d)\) be ranks (if d is not in list i, treat rank as \(\infty\) or a large constant). Let n be the number of lists (1, 2, or 3) containing d. Then:
  - **Modified RRF (3-way):** \(s(d) = (w_i + n/10) \cdot \sum_{i=1}^{3} 1/(k + r_i(d))\), with e.g. \(w_1 = w_2 = w_3 = 1\) or tuned (Novelty 1).
  - **Standard RRF (3-way):** \(s(d) = \sum_{i=1}^{3} 1/(k + r_i(d))\).
  - **CombSUM (3-way):** If using score-based fusion, normalise scores from all three runs and sum (or weighted sum). Rank by combined score.
  Compare 2-way (original + full-hypothesis) vs 3-way (original + full-hypothesis + third route) using the same fusion method. Optionally compare “half-hypothesis” vs “different-λ” as the third route.

- **Per-route weights (optional):** For 3-way, try different (w₁, w₂, w₃), e.g. (1, 1, 1), (1.2, 0.9, 0.9) to favour the original route, or (1, 1, 0.8) to down-weight the third route. Report whether tuning weights improves over equal weights.

- **Analysis:** Report MAP, nDCG@10, Recall@1000 for 2-way vs 3-way. Optionally, for each topic, note whether 3-way improved or degraded the metric; if possible, correlate with hypothesis length or query length to suggest when 3-way helps.

## 4.4 How it will make a difference

- If 3-way fusion beats 2-way on average, we improve metrics and show that the paper’s “three routes can help” finding holds when the third route is derived from the same hypothesis (no extra LLM cost).
- Comparing half-hypothesis vs different-λ as the third route can yield a recommendation for how to form the third run.
- If 2-way remains best, we still add a negative result: on this setup, a third route from the same hypothesis does not help, which is useful for practitioners and for positioning the paper’s two-route design.

## 4.5 Importance for the research contribution

- Directly extends the paper’s multi-route ablation in a cost-free way (same hypothesis).
- Addresses the question “can we get the benefit of a third route without more LLM calls?”
- 3-way fusion formulas (RRF, CombSUM) are natural extensions and can be compared (linking to Novelty 2).

## 4.6 Limitations in current (paper’s) implementation

- The paper does not test a third route that reuses the same hypothesis; their third route uses different LLM-generated content.
- No 3-way fusion formula or per-route weights for three lists.

## 4.7 Research limitations (our work)

- Only one (or two) way(s) to form the third route are tested; other variants (e.g. keyphrase-only from hypothesis) could be tried in future work.
- Generalisation to more than three routes (e.g. four from the same hypothesis) is not in scope; the paper already suggests diminishing returns and possible degradation.
- Single dataset; 3-way benefit may be dataset-dependent.

---

# Novelty 5: Deeper retrieval and score-based re-ranking

## 5.1 Why this novelty

The paper retrieves the top 1000 documents per route and fuses them; the final output is the fused ranking of the union of documents. Recall@1000 is evaluated over the top 1000 of this fused list. If we retrieve **more** than 1000 per route (e.g. 1500 or 2000), the fusion pool becomes larger, so more relevant documents may appear in the union and Recall@1000 can improve. Conversely, the paper does not re-rank the fused set: the final order is determined only by the fusion score. After fusion, we have a set of documents with two (or three) retrieval scores available. Re-ranking this set by the **original-query BM25 score** (or by a combination of the two BM25 scores) could better prioritise documents that are highly relevant to the original query while still benefiting from the expansion route’s recall. This novelty combines (1) deeper retrieval to potentially improve recall and (2) score-based re-ranking of the fused set to potentially improve precision at the top (nDCG@10, MAP).

## 5.2 What the base paper already does

The paper retrieves top 1000 per route and fuses; it does not experiment with retrieval depth or with re-ranking the fused list by relevance scores. So there is scope for novelty in both depth and re-ranking.

## 5.3 Approach in detail

- **Deeper retrieval:** For both routes (and, if using 3-way, for all three routes), retrieve top **2000** (or 1500) documents instead of 1000. Fuse these runs with the same fusion method as in the main experiments. From the fused list, take the **top 1000** for evaluation. Compare to the baseline where we retrieve 1000 per route and fuse. Metrics: MAP, nDCG@10, Recall@1000. Expectation: Recall@1000 may improve because the pool is larger; MAP and nDCG@10 may also improve if more relevant documents enter the fused top 1000.

- **Re-ranking after fusion:** Keep retrieval at 1000 per route (or at the same depth used in the main results). After fusion, we have a list of documents with fusion scores. For each document in this list, we also have (or can retain) its BM25 score from the **original-query** run (and optionally from the hypothesis run). Re-rank the **fused set** (top 1000 or the full union) by:
  - **Option A:** Original-query BM25 score only. Rationale: the original query is the user’s intent; ranking by it may put the most relevant documents first.
  - **Option B:** Sum (or weighted sum) of the two BM25 scores (original + hypothesis run). Rationale: documents that score well on both runs may be more reliably relevant.
  Produce the final ranking of 1000 documents and evaluate. Compare to the baseline fusion (no re-ranking). Expectation: nDCG@10 and MAP may improve if the re-ranking better orders the top documents.

- **Combination (optional):** Run “deeper retrieval” (e.g. 2000 per route) then fuse, then re-rank the fused top 1000 by original-query BM25 score, and evaluate. Report whether the combination beats both “depth only” and “re-rank only.”

- **Efficiency note:** Deeper retrieval doubles (or 1.5×) the number of retrieved documents per query; we do not optimise latency or indexing, but we can state the trade-off in the paper (e.g. “retrieving 2000 per route increases retrieval cost by X%”).

## 5.4 How it will make a difference

- **Deeper retrieval:** Can improve Recall@1000 and, if more relevant docs enter the fused top 1000, MAP and nDCG@10.
- **Re-ranking:** Can improve the ordering at the top (nDCG@10, MAP) by using the original-query (or combined) relevance score instead of only the fusion rank score.
- **Combined:** May give the best of both: larger pool and better ordering.

## 5.5 Importance for the research contribution

- Addresses two dimensions the paper does not explore: retrieval depth and post-fusion re-ranking.
- No extra LLM cost; only more retrieval (depth) and a different final ranking (re-rank).
- Re-ranking is a simple, interpretable idea that connects fusion to the original query’s relevance scores.
- Efficiency trade-off (depth) can be briefly discussed as a limitation or practical consideration.

## 5.6 Limitations in current (paper’s) implementation

- Fixed retrieval depth (1000); no ablation on depth.
- No re-ranking of the fused list by relevance scores; final ranking is fusion-only.

## 5.7 Research limitations (our work)

- Deeper retrieval increases computational cost; we do not optimise or minimise depth.
- Re-ranking uses only BM25 scores (no learned reranker); we state this as a limitation and possible future work.
- Single dataset; benefit of depth and re-ranking may vary across collections.

---

# Summary table: novelties at a glance

| Novelty | Why | How (short) | Expected impact | Main limitation (ours) |
|--------|-----|-------------|------------------|--------------------------|
| 1. Hyperparameter tuning | Paper fixes λ, k, w₁, w₂ | Grid over λ, k, (w₁,w₂); optional rule by query/hyp length | Better MAP/nDCG/R@1k; or evidence that default is robust | Single dataset; no learned adaptation |
| 2. Fusion ablation | Paper uses only modified RRF | Same 2 runs; compare modified RRF, standard RRF, CombSUM, score fusion | Best method identified; possible gain over paper’s fusion | One dataset; no learned fusion |
| 3. Hypothesis truncation | Full hypothesis may add noise | Use first 32/64/96 tokens (or 1–2 sentences); stratify by hyp length | Less noise, better precision; analysis of when it helps | Heuristic lengths; no quality model |
| 4. Three-way fusion | Paper shows 3 routes can help but uses extra LLM | Third route = same hyp (half or different λ); 3-way RRF/CombSUM | Possible gain over 2-way; no extra API cost | One third-route variant; single dataset |
| 5. Deeper retrieval + re-rank | Paper uses depth 1000 and no re-rank | Retrieve 2000/route and/or re-rank fused set by orig BM25 | Higher R@1k (depth); better nDCG/MAP (re-rank) | More retrieval cost; no learned reranker |

---

# Suggested order of implementation and reporting

1. **Novelty 1 (tuning):** Run grid; report best (λ, k, w₁, w₂) and sensitivity; optionally add adaptive rule and compare.
2. **Novelty 2 (fusion):** On the same two runs, compare all fusion methods; report best method and short discussion.
3. **Novelty 3 (truncation):** Run full vs truncated; report best truncation and, if possible, analysis by hypothesis length.
4. **Novelty 4 (3-way):** Add third route(s); run 2-way vs 3-way; report metrics and optional per-route weights.
5. **Novelty 5 (depth + re-rank):** Run depth ablation and re-rank ablation (and optionally combined); report metrics and one sentence on cost.

In the paper, present baselines (BM25, Exp4Fuse 2-way with paper settings), then each novelty with tables and short analysis, and a final table or section that summarises which extensions improve over the base Exp4Fuse and by how much. State limitations (single dataset, no extra LLM, heuristic/rule-based adaptations) in a dedicated subsection and in the conclusion.

---

*This document is the detailed research plan for the Exp4Fuse extension project. All five novelties are designed to be implementable without additional LLM API cost or new datasets, and to be reportable at research-paper level with clear contributions, ablations, and limitations.*
