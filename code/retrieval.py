#!/usr/bin/env python3
"""Retrieval utilities for Exp4Fuse experiments."""
import json
import os
import re
from typing import Callable, Dict, List, Tuple

RunByQid = Dict[str, List[Tuple[str, int, float]]]


def sanitize_hypothesis(text: str) -> str:
    """Normalize hypothesis text so BM25 sees plain content terms."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"\*\*?", "", t)
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"[#_\-]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def script_paths(script_dir: str):
    trec_data = os.path.join(script_dir, "TREC DL19 data")
    return {
        "topics_data": os.path.join(trec_data, "dl19_topics"),
        "hy_data": os.path.join(trec_data, "dl19_hypothesis"),
        "topics": os.path.join(script_dir, "dl19_topics"),
        "hy": os.path.join(script_dir, "dl19_hy"),
    }


def ensure_topics_and_hypothesis(script_dir: str):
    """Ensure local dl19_topics and dl19_hy exist in project root."""
    paths = script_paths(script_dir)
    if not os.path.isfile(paths["topics"]) and os.path.isfile(paths["topics_data"]):
        with open(paths["topics_data"]) as f:
            data = json.load(f)
        with open(paths["topics"], "w") as f:
            json.dump(data, f, indent=0)
    if not os.path.isfile(paths["hy"]):
        if not os.path.isfile(paths["hy_data"]):
            raise FileNotFoundError(
                "dl19_hy missing and source TREC DL19 data/dl19_hypothesis not found."
            )
        with open(paths["hy_data"]) as f:
            data = json.load(f)
        with open(paths["hy"], "w") as f:
            json.dump(data, f, indent=0)
    return paths["topics"], paths["hy"]


def load_topics_and_hy(script_dir: str):
    topics_path, hy_path = ensure_topics_and_hypothesis(script_dir)
    with open(topics_path) as f:
        topics = json.load(f)
    with open(hy_path) as f:
        hy_rows = json.load(f)
    return topics, hy_rows


def require_searcher():
    """Load LuceneSearcher with friendly Java-21 message if needed."""
    try:
        from pyserini.search.lucene import LuceneSearcher
    except Exception as e:
        err = str(e)
        if (
            "UnsupportedClassVersionError" in err
            or "class file version 65.0" in err
            or "versions up to 61.0" in err
            or "jdk.incubator.vector" in err
            or "FindException" in err
        ):
            raise RuntimeError(
                "Pyserini requires Java 21. Install openjdk-21-jdk and set java/javac."
            ) from e
        raise
    return LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")


def build_augmented_query(query: str, hypothesis: str, lambda_: int = 5) -> str:
    prefix = (query + ".") * lambda_
    if hypothesis:
        return f"{prefix} {hypothesis}"
    return prefix


def full_hypothesis(raw_hy: str) -> str:
    return sanitize_hypothesis(raw_hy)


def truncate_hypothesis(raw_hy: str, n_tokens: int) -> str:
    cleaned = sanitize_hypothesis(raw_hy)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    return " ".join(tokens[:n_tokens])


def first_half_hypothesis(raw_hy: str) -> str:
    cleaned = sanitize_hypothesis(raw_hy)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    half = max(1, len(tokens) // 2)
    return " ".join(tokens[:half])


def _search_as_run(searcher, qid: str, query: str, depth: int) -> List[Tuple[str, int, float]]:
    hits = searcher.search(query, k=depth)
    return [(hit.docid, rank, float(hit.score)) for rank, hit in enumerate(hits, 1)]


def build_original_run(searcher, topics, depth: int = 1000) -> RunByQid:
    run = {}
    for qid, query in topics:
        run[str(qid)] = _search_as_run(searcher, str(qid), query, depth)
    return run


def build_expansion_run(
    searcher,
    hy_rows,
    lambda_: int = 5,
    depth: int = 1000,
    hypothesis_builder: Callable[[str], str] = full_hypothesis,
) -> RunByQid:
    run = {}
    for qid, query, raw_hy in hy_rows:
        hy_txt = hypothesis_builder(raw_hy)
        aug_query = build_augmented_query(query, hy_txt, lambda_=lambda_)
        run[str(qid)] = _search_as_run(searcher, str(qid), aug_query, depth)
    return run


def write_run_file(run: RunByQid, path: str, runid: str = "rank", top_k: int = 1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for qid in sorted(run.keys(), key=lambda x: int(x)):
            for docid, rank, score in run[qid][:top_k]:
                f.write(f"{qid} Q0 {docid} {rank} {score} {runid}\n")

