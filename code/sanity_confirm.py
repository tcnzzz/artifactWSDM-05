"""Independent sanity check of per-cell means (no bootstrap, no library code paths).

Reads shard jsonl directly, restricts to goals_test positions 80..199, requires all three
depths per question, and prints recall@25, mean judge calls and record counts for each cell.
"""
import glob
import json
import os
from collections import defaultdict

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ART, "results", "runs", "test")

order, seen = [], set()
for line in open(os.path.join(ART, "data", "goals_test.jsonl")):
    g = json.loads(line)
    if g["qid"] not in seen:
        seen.add(g["qid"])
        order.append(g["qid"])
pos = {q: i for i, q in enumerate(order)}
conf_q = {q for q, i in pos.items() if 80 <= i < 200}
print("confirmatory questions:", len(conf_q))


def load(tag):
    rows = {}
    for f in glob.glob(os.path.join(RUNS, tag, "shard*.jsonl")):
        for line in open(f):
            j = json.loads(line)
            q = j["qid"]
            if q in conf_q:
                rows[j["gid"]] = j
    return rows


tags = ["index_r1_rho0.1", "ours_r1_rho0.1", "band_r1_rho0.1", "topm_r1_rho0.1", "thompson_r1_rho0.1",
        "ours_r1_rho1", "band_r1_rho1", "topm_r1_rho1", "thompson_r1_rho1",
        "index_r1_weak_rho0.1", "ours_r1_weak_rho0.1", "band_r1_weak_rho0.1", "topm_r1_weak_rho0.1",
        "thompson_r1_weak_rho0.1", "ours_r1_weak_rho1", "band_r1_weak_rho1", "topm_r1_weak_rho1",
        "thompson_r1_weak_rho1"]
R = {t: load(t) for t in tags}
for t, rows in R.items():
    if not rows:
        print(f"{t:26s} EMPTY")
        continue
    byq = defaultdict(set)
    for g, r in rows.items():
        byq[r["qid"]].add(r["depth"])
    full = {q for q, ds in byq.items() if ds == {2, 3, 4}}
    gids = [g for g, r in rows.items() if r["qid"] in full]
    rec = sum(rows[g]["gold@25"] / 2.0 for g in gids) / len(gids)
    calls = sum(rows[g]["n_llm"] for g in gids) / len(gids)
    print(f"{t:26s} q={len(full):3d} crawls={len(gids):4d} recall@25={rec:.6f} calls={calls:.3f}")

# paired per-question check for the two headline cells
for tag in ("ours_r1_weak_rho1", "index_r1_weak_rho0.1"):
    pass
a = R["ours_r1_weak_rho1"]; b = R["index_r1_weak_rho0.1"]
gids = sorted(set(a) & set(b))
d = [a[g]["gold@25"] / 2.0 - b[g]["gold@25"] / 2.0 for g in gids]
print(f"\nzero-shot rho=1: TwoCost-minus-scorer mean over {len(gids)} paired crawls = {sum(d)/len(d):+.6f}; "
      f"crawls with diff={sum(1 for x in d if x != 0)}")
a = R["ours_r1_rho0.1"]; b = R["index_r1_rho0.1"]
gids = sorted(set(a) & set(b))
d = [a[g]["gold@25"] / 2.0 - b[g]["gold@25"] / 2.0 for g in gids]
print(f"trained rho=0.1: TwoCost-minus-scorer mean over {len(gids)} paired crawls = {sum(d)/len(d):+.6f}; "
      f"crawls with diff={sum(1 for x in d if x != 0)}")
a = R["topm_r1_weak_rho0.1"]; b = R["index_r1_weak_rho0.1"]
gids = sorted(set(a) & set(b))
d = [a[g]["gold@25"] / 2.0 - b[g]["gold@25"] / 2.0 for g in gids]
print(f"zero-shot rho=0.1: topm-minus-scorer mean over {len(gids)} paired crawls = {sum(d)/len(d):+.6f}; "
      f"crawls with diff={sum(1 for x in d if x != 0)}")
