#!/usr/bin/env python3
"""Fusion and run parsing utilities for Exp4Fuse experiments."""
from collections import defaultdict
from typing import Dict, List, Tuple

RunByQid = Dict[str, List[Tuple[str, int, float]]]
DocStatsByQid = Dict[str, Dict[str, Dict[str, float]]]


def load_run_file(path: str) -> RunByQid:
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, docid, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
            run[qid].append((docid, rank, score))
    for qid in run:
        run[qid].sort(key=lambda x: x[1])
    return dict(run)


def run_to_docstats(run: RunByQid, run_name: str) -> DocStatsByQid:
    by_qid = defaultdict(dict)
    for qid, docs in run.items():
        for docid, rank, score in docs:
            by_qid[qid].setdefault(docid, {})
            by_qid[qid][docid][f"{run_name}_rank"] = rank
            by_qid[qid][docid][f"{run_name}_score"] = score
    return by_qid


def _collect_docstats(runs: List[RunByQid]) -> DocStatsByQid:
    merged = defaultdict(dict)
    for idx, run in enumerate(runs):
        name = f"r{idx+1}"
        part = run_to_docstats(run, name)
        for qid, docs in part.items():
            for docid, stats in docs.items():
                merged[qid].setdefault(docid, {})
                merged[qid][docid].update(stats)
    return merged


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 0.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def fuse_runs(
    runs: List[RunByQid],
    method: str = "modified_rrf",
    k: int = 60,
    weights: Tuple[float, ...] = (1.0, 1.0),
    alpha: float = 0.5,
    top_k: int = 1000,
) -> RunByQid:
    """Fuse 2 or 3 runs and return qid -> [(docid, rank, fused_score)]."""
    docstats = _collect_docstats(runs)
    out = {}

    for qid, docs in docstats.items():
        fused = {}
        num_runs = len(runs)
        w = list(weights)
        if len(w) < num_runs:
            w.extend([1.0] * (num_runs - len(w)))

        if method in {"comb_sum", "score_linear"}:
            norm_scores_per_run = []
            for i in range(num_runs):
                raw = {}
                key = f"r{i+1}_score"
                for docid, st in docs.items():
                    if key in st:
                        raw[docid] = st[key]
                norm_scores_per_run.append(_minmax_norm(raw))

        for docid, st in docs.items():
            present = 0
            if method == "modified_rrf":
                rank_sum = 0.0
                for i in range(num_runs):
                    key = f"r{i+1}_rank"
                    if key in st:
                        present += 1
                        rank_sum += (w[i] * (1.0 / (k + st[key])))
                fused[docid] = (1.0 + present / 10.0) * rank_sum
            elif method == "standard_rrf":
                rank_sum = 0.0
                for i in range(num_runs):
                    key = f"r{i+1}_rank"
                    if key in st:
                        rank_sum += 1.0 / (k + st[key])
                fused[docid] = rank_sum
            elif method == "comb_sum":
                score = 0.0
                for i in range(num_runs):
                    d = norm_scores_per_run[i].get(docid, 0.0)
                    score += d
                fused[docid] = score
            elif method == "score_linear":
                # Defined for 2-run setup only in this project.
                s1 = norm_scores_per_run[0].get(docid, 0.0)
                s2 = norm_scores_per_run[1].get(docid, 0.0)
                fused[docid] = alpha * s1 + (1.0 - alpha) * s2
            elif method == "borda":
                score = 0.0
                for i in range(num_runs):
                    key = f"r{i+1}_rank"
                    if key in st:
                        score += (top_k - st[key] + 1) if st[key] <= top_k else 0.0
                fused[docid] = score
            else:
                raise ValueError(f"Unsupported fusion method: {method}")

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out[qid] = [(docid, rank + 1, float(score)) for rank, (docid, score) in enumerate(ranked)]
    return out

