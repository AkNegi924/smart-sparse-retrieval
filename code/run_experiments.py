#!/usr/bin/env python3
"""Experiment runner for Exp4Fuse course-project novelties 1-3."""
import argparse
import csv
import os
from functools import partial

from eval_util import evaluate_runs, run_trec_eval
from fusion import fuse_runs
from retrieval import (
    build_expansion_run,
    build_original_run,
    full_hypothesis,
    load_topics_and_hy,
    require_searcher,
    truncate_hypothesis,
    write_run_file,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _ensure_dirs():
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _materialize_run(run_obj, file_name):
    path = os.path.join(RUNS_DIR, file_name)
    write_run_file(run_obj, path)
    return path


def run_baseline(searcher, topics, hy_rows):
    original_run = build_original_run(searcher, topics, depth=1000)
    hy_run = build_expansion_run(searcher, hy_rows, lambda_=5, depth=1000, hypothesis_builder=full_hypothesis)

    p_ori = _materialize_run(original_run, "dl19_bm25_original_baseline")
    p_hy = _materialize_run(hy_run, "dl19_bm25_hy_lambda5_full")
    fused = fuse_runs([original_run, hy_run], method="modified_rrf", k=60, weights=(1.0, 1.0))
    p_fused = _materialize_run(fused, "dl19_bm25_exp4fuse_baseline")

    rows = evaluate_runs(
        {
            "baseline_original": p_ori,
            "baseline_hypothesis_only": p_hy,
            "baseline_exp4fuse": p_fused,
        }
    )
    _write_csv(os.path.join(RESULTS_DIR, "baseline_parity.csv"), rows)
    return original_run, hy_run, rows


def novelty1_reduced_grid(searcher, topics, hy_rows):
    rows = []
    original_run = build_original_run(searcher, topics, depth=1000)
    original_path = _materialize_run(original_run, "nov1_original")

    lambda_grid = [4, 5, 6]
    k_grid = [30, 60]
    weight_grid = [(1.0, 1.0), (1.2, 0.8)]

    hy_runs = {}
    for lam in lambda_grid:
        hy_run = build_expansion_run(searcher, hy_rows, lambda_=lam, depth=1000, hypothesis_builder=full_hypothesis)
        hy_runs[lam] = hy_run
        _materialize_run(hy_run, f"nov1_hy_lambda{lam}")

    for lam in lambda_grid:
        for k in k_grid:
            for w1, w2 in weight_grid:
                fused = fuse_runs(
                    [original_run, hy_runs[lam]],
                    method="modified_rrf",
                    k=k,
                    weights=(w1, w2),
                )
                name = f"nov1_fused_l{lam}_k{k}_w{w1}_{w2}"
                fused_path = _materialize_run(fused, name)
                metrics = run_trec_eval(fused_path)
                rows.append(
                    {
                        "experiment": name,
                        "lambda": lam,
                        "k": k,
                        "w1": w1,
                        "w2": w2,
                        "map": metrics.get("map"),
                        "ndcg_cut_10": metrics.get("ndcg_cut_10"),
                        "recall_1000": metrics.get("recall_1000"),
                        "original_run": original_path,
                        "fused_run": fused_path,
                    }
                )

    _write_csv(os.path.join(RESULTS_DIR, "novelty1_grid.csv"), rows)
    return rows


def novelty2_fusion_ablation(searcher, topics, hy_rows):
    original_run = build_original_run(searcher, topics, depth=1000)
    hy_run = build_expansion_run(searcher, hy_rows, lambda_=5, depth=1000, hypothesis_builder=full_hypothesis)
    rows = []

    methods = [
        ("modified_rrf", {}),
        ("standard_rrf", {}),
        ("score_linear", {"alpha": 0.5}),
        ("comb_sum", {}),
    ]

    for method, opts in methods:
        fused = fuse_runs([original_run, hy_run], method=method, k=60, weights=(1.0, 1.0), **opts)
        name = f"nov2_{method}" + (f"_a{opts['alpha']}" if "alpha" in opts else "")
        path = _materialize_run(fused, name)
        metrics = run_trec_eval(path)
        rows.append(
            {
                "experiment": name,
                "method": method,
                "alpha": opts.get("alpha"),
                "map": metrics.get("map"),
                "ndcg_cut_10": metrics.get("ndcg_cut_10"),
                "recall_1000": metrics.get("recall_1000"),
                "fused_run": path,
            }
        )

    _write_csv(os.path.join(RESULTS_DIR, "novelty2_methods.csv"), rows)
    return rows


def novelty3_truncation(searcher, topics, hy_rows):
    original_run = build_original_run(searcher, topics, depth=1000)
    rows = []

    variants = [
        ("full", full_hypothesis),
        ("trunc64", partial(truncate_hypothesis, n_tokens=64)),
        ("trunc96", partial(truncate_hypothesis, n_tokens=96)),
    ]

    for label, hy_fn in variants:
        hy_run = build_expansion_run(searcher, hy_rows, lambda_=5, depth=1000, hypothesis_builder=hy_fn)
        fused = fuse_runs([original_run, hy_run], method="modified_rrf", k=60, weights=(1.0, 1.0))
        path = _materialize_run(fused, f"nov3_fused_{label}")
        metrics = run_trec_eval(path)
        rows.append(
            {
                "experiment": f"nov3_{label}",
                "truncation": label,
                "map": metrics.get("map"),
                "ndcg_cut_10": metrics.get("ndcg_cut_10"),
                "recall_1000": metrics.get("recall_1000"),
                "fused_run": path,
            }
        )

    _write_csv(os.path.join(RESULTS_DIR, "novelty3_truncation.csv"), rows)
    return rows


def print_rows(title, rows):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    for row in rows:
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Run Exp4Fuse novelty experiments (1-3).")
    parser.add_argument(
        "--novelty",
        choices=["baseline", "1", "2", "3", "all"],
        default="all",
        help="Which experiment set to run.",
    )
    args = parser.parse_args()

    _ensure_dirs()
    topics, hy_rows = load_topics_and_hy(SCRIPT_DIR)
    searcher = require_searcher()

    if args.novelty in {"baseline", "all"}:
        _, _, rows = run_baseline(searcher, topics, hy_rows)
        print_rows("Baseline parity", rows)
    if args.novelty in {"1", "all"}:
        rows = novelty1_reduced_grid(searcher, topics, hy_rows)
        print_rows("Novelty 1 grid", rows)
    if args.novelty in {"2", "all"}:
        rows = novelty2_fusion_ablation(searcher, topics, hy_rows)
        print_rows("Novelty 2 methods", rows)
    if args.novelty in {"3", "all"}:
        rows = novelty3_truncation(searcher, topics, hy_rows)
        print_rows("Novelty 3 truncation", rows)

    print(f"\nSaved run files under: {RUNS_DIR}")
    print(f"Saved result tables under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

