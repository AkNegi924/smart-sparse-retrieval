#!/usr/bin/env python3
"""
Generate TREC DL19 hypothesis documents using OpenRouter (openai/gpt-oss-120b:free).
Uses the paper's TREC DL19 prompt and test file (dl19_topics). Saves dl19_hy for the pipeline.
"""
import json
import os
import sys

# Script dir = Exp4Fuse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Load .env from Exp4Fuse or parent (SSD)
for _dir in (SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)):
    _env = os.path.join(_dir, ".env")
    if os.path.isfile(_env):
        with open(_env) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"").strip()
                    if k:
                        os.environ.setdefault(k, v)
        break

if not os.environ.get("OPENROUTER_API_KEY"):
    print("Error: OPENROUTER_API_KEY not set. Add it to .env in Exp4Fuse or SSD.")
    sys.exit(1)

from openai import OpenAI

# Paper: TREC DL19 prompt (Exp4Fuse/prompt template.txt)
# Ask for plain text so BM25 sees passage-like terms (no markdown/headers/tables)
SYSTEM_PROMPT = "Please write a short passage to answer the question. Use plain text only, no markdown or formatting."
# Paper section 4.1: temperature 0.6, top_p 0.9, max 128 tokens
MODEL = "openai/gpt-oss-120b:free"
MAX_TOKENS = 128
TEMPERATURE = 0.6
TOP_P = 0.9

TREC_DATA = os.path.join(SCRIPT_DIR, "TREC DL19 data")
TOPICS_PATH = os.path.join(TREC_DATA, "dl19_topics")
HYPOTHESIS_PATH = os.path.join(TREC_DATA, "dl19_hypothesis")
DL19_HY_PATH = os.path.join(SCRIPT_DIR, "dl19_hy")  # notebook expects this in cwd


def format_error(e):
    err_str = str(e).lower()
    msg = str(e)
    if hasattr(e, "response") and e.response is not None:
        try:
            body = e.response.json()
            msg = body.get("error", {}).get("message", msg)
        except Exception:
            pass
        if e.response.status_code == 401:
            msg = "Invalid API key. Check OPENROUTER_API_KEY in .env."
        elif e.response.status_code == 429:
            msg = "Rate limit. Try again later."
    return msg


def main():
    with open(TOPICS_PATH) as f:
        topics = json.load(f)
    # topics: list of [qid, query]
    if not topics:
        print("No topics in", TOPICS_PATH)
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    dl19_hy = []
    for i, item in enumerate(topics):
        qid, query = item[0], item[1]
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            text = (r.choices[0].message.content or "").strip()
            dl19_hy.append([qid, query, text])
            print(f"  [{i+1}/{len(topics)}] qid={qid} len(hy)={len(text)}")
        except Exception as e:
            print("API Error:", format_error(e))
            sys.exit(1)

    for path in (HYPOTHESIS_PATH, DL19_HY_PATH):
        with open(path, "w") as f:
            json.dump(dl19_hy, f, indent=0)
        print("Wrote", path)
    print("Done. Run the pipeline from Exp4Fuse: python run_exp4fuse_dl19.py")


if __name__ == "__main__":
    main()
