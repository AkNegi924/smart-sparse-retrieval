#!/usr/bin/env python3
"""
Run Exp4Fuse pipeline on TREC DL19: BM25 (original + hypothesis routes), RRF fusion, trec_eval.
Uses paper test files: dl19_topics, dl19_hy. Run from Exp4Fuse dir (or we set cwd).
Requires JDK 21 (Pyserini's JARs are built with Java 21). Install openjdk-21-jdk
and ensure java/javac point to it (e.g. sudo update-alternatives --config java).
"""
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

TREC_DATA = os.path.join(SCRIPT_DIR, "TREC DL19 data")
TOPICS_PATH = os.path.join(TREC_DATA, "dl19_topics")
HYPOTHESIS_PATH = os.path.join(TREC_DATA, "dl19_hypothesis")
DL19_TOPICS_FILE = os.path.join(SCRIPT_DIR, "dl19_topics")
DL19_HY_FILE = os.path.join(SCRIPT_DIR, "dl19_hy")

# Ensure notebook-style files exist in Exp4Fuse
if not os.path.isfile(DL19_TOPICS_FILE) and os.path.isfile(TOPICS_PATH):
    with open(TOPICS_PATH) as f:
        data = json.load(f)
    with open(DL19_TOPICS_FILE, "w") as f:
        json.dump(data, f, indent=0)
    print("Copied dl19_topics to Exp4Fuse/")
if not os.path.isfile(DL19_HY_FILE):
    if os.path.isfile(HYPOTHESIS_PATH):
        with open(HYPOTHESIS_PATH) as f:
            data = json.load(f)
        with open(DL19_HY_FILE, "w") as f:
            json.dump(data, f, indent=0)
        print("Copied dl19_hypothesis -> dl19_hy in Exp4Fuse/")
    else:
        print("Error: dl19_hy not found. Run: python generate_dl19_hypothesis.py")
        sys.exit(1)

with open(DL19_TOPICS_FILE) as f:
    dl19_topics = json.load(f)
with open(DL19_HY_FILE) as f:
    dl19_hy = json.load(f)

# Paper expects passage-like text; strip markdown/formatting so BM25 sees content terms only
def sanitize_hypothesis(text):
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"\*\*?", "", t)  # ** and *
    t = re.sub(r"\|", " ", t)   # table pipes
    t = re.sub(r"\n+", " ", t)  # newlines -> space
    t = re.sub(r"[#_\-]{2,}", " ", t)  # headers/underlines
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Pyserini retrieval (requires JDK 21 — Anserini JARs use class file version 65.0)
try:
    from pyserini.search.lucene import LuceneSearcher
except Exception as e:
    err = str(e)
    if "UnsupportedClassVersionError" in err or "class file version 65.0" in err or "versions up to 61.0" in err:
        print("Java version error: Pyserini's JARs require JDK 21 (you have 17). Install and select Java 21:")
        print("  sudo apt install openjdk-21-jdk")
        print("  sudo update-alternatives --config java   # choose java-21")
        print("  sudo update-alternatives --config javac")
        sys.exit(1)
    if "jdk.incubator.vector" in err or "FindException" in err or ("Module" in err and "not found" in err):
        print("Java module error: Pyserini needs JDK 21. Install: sudo apt install openjdk-21-jdk")
        sys.exit(1)
    raise

print("Loading BM25 index (msmarco-v1-passage)...")
searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")

# Original query run
print("BM25 original query run -> dl19_bm25")
with open("dl19_bm25", "w") as f:
    for i in range(len(dl19_topics)):
        qid = dl19_topics[i][0]
        question = dl19_topics[i][1]
        hits = searcher.search(question, k=1000)
        for rank, hit in enumerate(hits, 1):
            f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score} rank\n")

# Hypothesis-augmented run (paper: qe = concat(qo x λ, rq), λ=5; notebook does (query+'.')*5 + hy)
# Use sanitized passage-like text so BM25 matches content; fallback to query-only if hypothesis empty
print("BM25 hypothesis-augmented run -> dl19_bm25_hy")
with open("dl19_bm25_hy", "w") as f:
    for i in range(len(dl19_hy)):
        qid = dl19_hy[i][0]
        raw_hy = dl19_hy[i][2]
        hy_clean = sanitize_hypothesis(raw_hy)
        if hy_clean:
            question = (dl19_hy[i][1] + ".") * 5 + " " + hy_clean
        else:
            question = (dl19_hy[i][1] + ".") * 5
        hits = searcher.search(question, k=1000)
        for rank, hit in enumerate(hits, 1):
            f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score} rank\n")

# Load run files for fusion
with open("dl19_bm25_hy") as file:
    hy1 = file.readlines()
hy = [[int(x.split()[0]), x.split()[1], int(x.split()[2]), int(x.split()[3]), x.split()[4], x.split()[5]] for x in hy1]
with open("dl19_bm25") as file:
    ori1 = file.readlines()
ori = [[int(x.split()[0]), x.split()[1], int(x.split()[2]), int(x.split()[3]), x.split()[4], x.split()[5]] for x in ori1]

# Paper: k=60, w1=w2=1; FR_score = (w_i + n/10) * sum(1/(k+r_i))
ratio = 60
print("Fusion (RRF k=60) -> dl19_bm25_Exp4Fuse")
with open("dl19_bm25_Exp4Fuse", "w") as f:
    for i in range(len(dl19_topics)):
        dictTem = {}
        for j in range(len(hy)):
            if dl19_topics[i][0] == hy[j][0]:
                try:
                    dictTem[hy[j][2]]["score1"] = 1 / (ratio + hy[j][3])
                except KeyError:
                    dictTem[hy[j][2]] = {}
                    dictTem[hy[j][2]]["score1"] = 1 / (ratio + hy[j][3])
        for j in range(len(ori)):
            if dl19_topics[i][0] == ori[j][0]:
                try:
                    dictTem[ori[j][2]]["score3"] = 1 / (ratio + ori[j][3])
                except KeyError:
                    dictTem[ori[j][2]] = {}
                    dictTem[ori[j][2]]["score3"] = 1 / (ratio + ori[j][3])
        for passage in dictTem.keys():
            num = len(dictTem[passage]) / 10 + 1
            score_sum = sum(dictTem[passage].values())
            dictTem[passage] = num * score_sum
        rank = 0
        for key, value in sorted(dictTem.items(), key=lambda item: item[1], reverse=True):
            rank += 1
            if rank <= 1000:
                topicnum = dl19_topics[i][0]
                f.write(f"{topicnum} Q0 {key} {rank} {value} rank\n")

# Evaluate with jtreceval JAR (paper metrics: MAP, nDCG@10, Recall@1k)
def _parse_trec_eval_stdout(stdout):
    """Parse trec_eval output lines: metric\tall\tvalue -> dict."""
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


def run_trec_eval(run_file):
    """Run jtreceval JAR for map, ndcg_cut.10, recall.1000."""
    from pyserini.util import download_evaluation_script
    from pyserini.search._base import get_qrels_file
    jar_path = download_evaluation_script("trec_eval", verbose=False)
    qrels_path = get_qrels_file("dl19-passage")
    run_path = os.path.join(SCRIPT_DIR, run_file)
    cmd = [
        "java", "-jar", jar_path,
        "-c", "-l", "2",
        "-m", "map", "-m", "ndcg_cut.10", "-m", "recall.1000",
        qrels_path, run_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=60)
    return _parse_trec_eval_stdout(result.stdout)

# Paper metrics order: MAP, nDCG@10, Recall@1k (keys from Java trec_eval: map, ndcg_cut_10, recall_1000)
def print_metrics(metrics, empty_msg="(no metrics)"):
    if not metrics:
        print(f"  {empty_msg}")
        return
    for key in ["map", "ndcg_cut_10", "recall_1000"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

print("\n--- TREC DL19 evaluation (paper metrics) ---")
print("BM25 (original):")
bm25_metrics = run_trec_eval("dl19_bm25")
print_metrics(bm25_metrics)
print("BM25 (hypothesis-augmented only):")
hy_metrics = run_trec_eval("dl19_bm25_hy")
print_metrics(hy_metrics)
print("BM25 + Exp4Fuse:")
fuse_metrics = run_trec_eval("dl19_bm25_Exp4Fuse")
print_metrics(fuse_metrics)
print("\nDone.")
