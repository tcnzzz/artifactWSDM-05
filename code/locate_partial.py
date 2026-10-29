"""Locate the two incomplete band-cascade questions and check block membership.

Reads results/runs/test/band_r1_rho0.1 and goals_test.jsonl; prints the position of every
question that does not have all three depths, and the block each belongs to.
"""
import json
import os
from collections import defaultdict

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ART, "results", "runs", "test")

order = []
seen = set()
for line in open(os.path.join(ART, "data", "goals_test.jsonl")):
    g = json.loads(line)
    if g["qid"] not in seen:
        seen.add(g["qid"])
        order.append(g["qid"])
pos = {q: i for i, q in enumerate(order)}

byq = defaultdict(list)
for name in os.listdir(os.path.join(RUNS, "band_r1_rho0.1")):
    if name.startswith("shard") and name.endswith(".jsonl"):
        for line in open(os.path.join(RUNS, "band_r1_rho0.1", name)):
            r = json.loads(line)
            byq[r["qid"]].append(r["depth"])

print("questions in band_r1_rho0.1:", len(byq))
for q, ds in sorted(byq.items(), key=lambda kv: pos.get(kv[0], 10**9)):
    if sorted(ds) != [2, 3, 4]:
        print(f"  incomplete qid={q} pos={pos.get(q)} depths={sorted(ds)} block={40 * (pos.get(q, 0) // 40)}-{40 * (pos.get(q, 0) // 40) + 40}")
print("positions range of all band questions:",
      min(pos[q] for q in byq), max(pos[q] for q in byq))
print("questions in block 200-240 present in band:", sum(1 for q in byq if 200 <= pos[q] < 240))
