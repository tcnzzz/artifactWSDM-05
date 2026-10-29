"""Pick top-m and Thompson hyperparameters on validation recall at the headline budget (never on test)."""
import os, json, glob, numpy as np
from common import R
from crawl import MODELS
from stats import load
B = 25; out = {}
for pol, key, grid in (("topm", lambda m, u: {"m": m, "mu": u}, [(1, 0), (3, 1), (5, 2)]), ("thompson", lambda k: {"ts_kappa": k}, [(1,), (3,), (10,)])):
    best = None; res = {}
    for cfg in grid:
        tag = f"_m{cfg[0]}u{cfg[1]}" if pol == "topm" else f"_k{cfg[0]}"; rows = load("val", f"{pol}{tag}_rho0.1")
        if not rows: continue
        rec = float(np.mean([r[f"gold@{B}"] / 2 for r in rows.values()])); calls = float(np.mean([r[f"llm@{B}"] for r in rows.values()])); res[tag] = dict(recall=rec, calls=calls, n=len(rows))
        if best is None or rec > best[0]: best = (rec, key(*cfg))
    out[pol] = dict(best=best[1], grid=res); json.dump(best[1], open(os.path.join(MODELS, f"{pol}.json"), "w"))
for pol in ("ours", "always"):
    rows = load("val", f"{pol}_rho0.1")
    if rows: out[pol] = dict(recall=float(np.mean([r[f"gold@{B}"] / 2 for r in rows.values()])), calls=float(np.mean([r[f"llm@{B}"] for r in rows.values()])), n=len(rows))
json.dump(out, open(os.path.join(R, "tables", "val_hparams.json"), "w"), indent=1); print(json.dumps(out, indent=1))
