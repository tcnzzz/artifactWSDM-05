"""Loading of run files and cluster bootstrap over questions."""
import os, json, glob
import numpy as np
from common import R
def load(split, name):
    rows = {}
    for f in sorted(glob.glob(os.path.join(R, "runs", split, name, "shard*.jsonl"))):
        for l in open(f):
            j = json.loads(l); rows[j["gid"]] = j
    return rows
def recall(rows, B, gids=None, key="gold"):
    gids = gids or sorted(rows); return np.array([rows[g][f"{key}@{B}"] / (2.0 if key == "gold" else 1.0) for g in gids])
def boot(values, qids, n=2000, seed=0):
    """Mean and 95% percentile interval, resampling questions (all depths of a question move together)."""
    values = np.asarray(values, float); uq, inv = np.unique(qids, return_inverse=True); rng = np.random.default_rng(seed)
    sums = np.bincount(inv, values, len(uq)); cnt = np.bincount(inv, minlength=len(uq)).astype(float)
    idx = rng.integers(0, len(uq), (n, len(uq))); m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(values.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
