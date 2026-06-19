# Exp4Fuse Reproduction Guide (Team Setup)

This guide helps a teammate set up the project in a fresh virtual environment and reproduce the same experiment outputs.

---

## 1) Prerequisites

## 1.1 System packages

- Python 3.10+ (tested with Python 3.12)
- Java 21 (required by Pyserini / Anserini)

Ubuntu/Debian example:

```bash
sudo apt update
sudo apt install -y python3 python3-venv openjdk-21-jdk
```

Verify Java:

```bash
java -version
javac -version
```

If Java is not 21, switch alternatives:

```bash
sudo update-alternatives --config java
sudo update-alternatives --config javac
```

---

## 2) Project setup in virtual environment

From the repo root:

```bash
cd /home/anush/Coding/SSD/Exp4Fuse
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes the core packages required for this project workflow.
- First-time installation of `pyserini` can take time and download many dependencies.

---

## 3) Data files expected

These files must exist (already present in this project):

- `dl19_topics`
- `dl19_hy`

Fallback sources are also supported by the code:

- `TREC DL19 data/dl19_topics`
- `TREC DL19 data/dl19_hypothesis`

If `dl19_topics` / `dl19_hy` are missing, the code auto-copies from the fallback paths.

---

## 4) Reproduce the baseline and all implemented novelties

Run each experiment separately:

```bash
source /home/anush/Coding/SSD/Exp4Fuse/.venv/bin/activate
python /home/anush/Coding/SSD/Exp4Fuse/run_experiments.py --novelty baseline
python /home/anush/Coding/SSD/Exp4Fuse/run_experiments.py --novelty 1
python /home/anush/Coding/SSD/Exp4Fuse/run_experiments.py --novelty 2
python /home/anush/Coding/SSD/Exp4Fuse/run_experiments.py --novelty 3
```

Or run everything in one command:

```bash
source /home/anush/Coding/SSD/Exp4Fuse/.venv/bin/activate
python /home/anush/Coding/SSD/Exp4Fuse/run_experiments.py --novelty all
```

---

## 5) What gets generated

## 5.1 Run files

Generated in:

- `runs/`

These are TREC-format files:

```text
qid Q0 docid rank score runid
```

## 5.2 Result tables (CSV)

Generated in:

- `results/baseline_parity.csv`
- `results/novelty1_grid.csv`
- `results/novelty2_methods.csv`
- `results/novelty3_truncation.csv`

---

## 6) Implemented novelty scope (for teammate context)

This code reproduces the **course-project scope**:

- Novelty 1: Hyperparameter tuning (reduced grid)
- Novelty 2: Fusion method ablation (core methods)
- Novelty 3: Hypothesis truncation (full / 64 / 96)

Deferred in current code path:

- Novelty 4 (three-way fusion)
- Novelty 5 (deeper retrieval + re-ranking)

---

## 7) Quick validation checklist

After running:

1. Confirm 4 CSV files exist in `results/`.
2. Confirm multiple run files exist in `runs/`.
3. Open `results/baseline_parity.csv` and verify that `baseline_original` and `baseline_exp4fuse` rows are present.
4. Open `results/novelty1_grid.csv` and verify it has multiple parameter combinations.
5. Open `results/novelty2_methods.csv` and verify rows for:
   - `modified_rrf`
   - `standard_rrf`
   - `score_linear`
   - `comb_sum`
6. Open `results/novelty3_truncation.csv` and verify rows for:
   - `full`
   - `trunc64`
   - `trunc96`

---

## 8) Troubleshooting

## 8.1 `ModuleNotFoundError: pyserini`

Make sure virtualenv is active and dependencies installed:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 8.2 Java class version / incubator module errors

Ensure Java 21 is active:

```bash
java -version
javac -version
```

Switch to Java 21 if needed:

```bash
sudo update-alternatives --config java
sudo update-alternatives --config javac
```

## 8.3 qrels/trec_eval download issue

The first run downloads qrels and `jtreceval` through Pyserini. Ensure internet access is available for those downloads.

---

## 9) Optional: hypothesis generation (not required for reproduction)

If you ever need to regenerate hypotheses:

```bash
source .venv/bin/activate
python generate_dl19_hypothesis.py
```

Requires:

- valid `OPENROUTER_API_KEY` in `.env`

For this project reproduction, **do not regenerate** hypotheses; use existing `dl19_hy`.

---

## 10) Recommended handoff package to teammate

Share this folder with:

- code files (`run_experiments.py`, `retrieval.py`, `fusion.py`, `eval_util.py`)
- data files (`dl19_topics`, `dl19_hy`)
- `requirements.txt`
- this guide (`REPRODUCTION_INSTRUCTIONS.md`)
- report (`ENDSEM_PROJECT_REPORT.md`)

Then your teammate can run the same commands and obtain matching CSV outputs.

