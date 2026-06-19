#!/usr/bin/env python3
"""Evaluation helpers for trec_eval via Pyserini."""
import subprocess


def _parse_trec_eval_stdout(stdout: str):
    out = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("runid"):
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) >= 3 and parts[1] == "all":
            try:
                out[parts[0]] = float(parts[2])
            except ValueError:
                pass
    return out


def run_trec_eval(run_path: str):
    from pyserini.util import download_evaluation_script
    from pyserini.search._base import get_qrels_file

    jar_path = download_evaluation_script("trec_eval", verbose=False)
    qrels_path = get_qrels_file("dl19-passage")
    cmd = [
        "java",
        "-jar",
        jar_path,
        "-c",
        "-l",
        "2",
        "-m",
        "map",
        "-m",
        "ndcg_cut.10",
        "-m",
        "recall.1000",
        qrels_path,
        run_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return _parse_trec_eval_stdout(result.stdout)


def evaluate_runs(label_to_path):
    rows = []
    for label, path in label_to_path.items():
        metrics = run_trec_eval(path)
        rows.append(
            {
                "experiment": label,
                "map": metrics.get("map"),
                "ndcg_cut_10": metrics.get("ndcg_cut_10"),
                "recall_1000": metrics.get("recall_1000"),
            }
        )
    return rows

