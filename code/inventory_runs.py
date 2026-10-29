#!/usr/bin/env python3
"""Inventory of saved test-run outputs per (policy, scorer, rho, depth).

Reads the shard jsonl files under results/runs/test/<tag>/ and reports, for every
run directory, the number of crawl records, the number of distinct questions, and
the per-depth counts. Adds a confirmatory-only view restricted to questions at
positions 80..199 of goals_test.jsonl (blocks the progress log records as done).

Writes results/tables/inventory.json. Read-only with respect to existing results.
"""
import json
import os
import sys
from collections import defaultdict, Counter

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ART, "results", "runs", "test")
OUT = os.path.join(ART, "results", "tables", "inventory.json")


def goal_order():
    ids = []
    with open(os.path.join(ART, "data", "goals_test.jsonl")) as f:
        for line in f:
            g = json.loads(line)
            q = g.get("qid") or g.get("id")
            if q not in ids:
                ids.append(q)
    return ids


def load_dir(path):
    recs = []
    for name in sorted(os.listdir(path)):
        if not name.startswith("shard") or not name.endswith(".jsonl"):
            continue
        with open(os.path.join(path, name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def summarize(recs, pos):
    if not recs:
        return {"records": 0, "questions": 0, "depths": {}, "qpos_min": None,
                "qpos_max": None, "in_block_80_200": 0, "in_block_80_200_q": 0}
    qs = {r["qid"] for r in recs}
    depths = Counter(r["depth"] for r in recs)
    ps = [pos.get(q) for q in qs if q in pos]
    inb = [r for r in recs if pos.get(r["qid"]) is not None and 80 <= pos[r["qid"]] < 200]
    # completeness: a question counts only if all three constructed depths are present
    byq = defaultdict(set)
    for r in recs:
        byq[r["qid"]].add(r["depth"])
    complete_q = sum(1 for q, ds in byq.items() if ds == {2, 3, 4})
    inb_q = {r["qid"] for r in inb}
    inb_complete = sum(1 for q in inb_q if byq[q] == {2, 3, 4})
    return {"records": len(recs), "questions": len(qs),
            "questions_all_three_depths": complete_q,
            "depths": dict(sorted(depths.items())),
            "qpos_min": min(ps) if ps else None, "qpos_max": max(ps) if ps else None,
            "in_block_80_200": len(inb),
            "in_block_80_200_q": len(inb_q),
            "in_block_80_200_q_all_three": inb_complete}


def main():
    order = goal_order()
    pos = {q: i for i, q in enumerate(order)}
    report = {"n_questions_in_goals_test": len(order), "runs": {}}
    for tag in sorted(os.listdir(RUNS)):
        path = os.path.join(RUNS, tag)
        if not os.path.isdir(path):
            continue
        recs = load_dir(path)
        if not recs:
            continue
        report["runs"][tag] = summarize(recs, pos)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {OUT}")
    print(f"{'tag':34s} {'records':>7s} {'q':>4s} {'q3d':>4s} {'in80-200':>8s} {'q80-200':>7s} {'q3d':>4s}  depths")
    for tag, s in report["runs"].items():
        print(f"{tag:34s} {s['records']:7d} {s['questions']:4d} {s['questions_all_three_depths']:4d} "
              f"{s['in_block_80_200']:8d} {s['in_block_80_200_q']:7d} {s['in_block_80_200_q_all_three']:4d}  {s['depths']}")


if __name__ == "__main__":
    main()
