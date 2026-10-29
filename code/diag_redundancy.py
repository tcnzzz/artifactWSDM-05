"""Diagnostic: how many alternative routes to a gold page does a crawl's frontier hold? For test crawls of the scorer-only
index policy (rows test_index_T and test_index_weakT), count distinct frontier pages at each distance from the nearest gold
page among links seen within the first 25 fetches. Writes results/tables/diag_redundancy.json."""
import os, glob, pickle, json, numpy as np
from common import D, R, load_goals
goals = {g["gid"]: g for g in load_goals("test")}; out = {}
for kind in ("index_T", "index_weakT"):
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "rows", f"test_{kind}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
    cnt = []
    for r in rows:
        m = r["in_crawl"]; allowed = set([goals[r["gid"]]["seed"]] + [int(x) for x in r["fetched"][:25]])
        sel = np.array([p in allowed for p in r["par"][m]]); y = r["y"][m][sel]; cnt.append(np.bincount(np.minimum(y, 3), minlength=4))
    cnt = np.array(cnt); out[kind] = dict(n_crawls=len(cnt), mean_links_seen=float(cnt.sum(1).mean()), mean_by_layer={str(k): float(cnt[:, k].mean()) for k in range(4)},
                                           frac_crawls_ge2_layer1=float((cnt[:, 1] >= 2).mean()), median_layer1=float(np.median(cnt[:, 1])), median_layer2=float(np.median(cnt[:, 2])))
json.dump(out, open(os.path.join(R, "tables", "diag_redundancy.json"), "w"), indent=1); print(json.dumps(out, indent=1))
