"""Read-only snapshot of the judge response cache used for the paper's judge-call macros.

`results/llm_usage.json` was written by `analyze.py` before the confirmatory crawls were judged
(76,246 rows). The paper's `NJudgeCalls` (125,278) and `NJudgePromptTok` (21.3) are the final cache
counts; this script records that final count without overwriting any saved file, and without any
write to the cache (SQLite is opened in read-only mode).
"""
import hashlib
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "cache", "llm.sqlite")
OUT = os.path.join(ROOT, "results", "llm_usage_final.json")

con = sqlite3.connect("file:" + os.path.abspath(DB) + "?mode=ro", uri=True)
n, t = con.execute("SELECT count(*), sum(ptok) FROM insp").fetchone()
con.close()

h = hashlib.sha256()
with open(DB, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)

rec = {
    "distinct_judge_calls": int(n),
    "prompt_tokens": int(t or 0),
    "model": "qwen2.5:7b",
    "table": "insp",
    "covers": ("distinct (model, question, parent, target) judgements cached across every run in the "
               "artifact, exploration and confirmatory, reused by all policies"),
    "note": ("supersedes results/llm_usage.json (76,246 rows), which analyze.py wrote before the "
             "confirmatory crawls were judged; the saved numbers.tex already carries these final counts"),
    "cache_sha256": h.hexdigest(),
}
with open(OUT, "w") as f:
    json.dump(rec, f, indent=1)
    f.write("\n")
print(json.dumps(rec, indent=1))
