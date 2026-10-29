"""Budget price lambda(B) = median index of the last page fetched within budget B by the encoder-only index policy on
validation goals (a measured shadow price, not tuned on recall). Writes data/models/lambda.json."""
import os, json, numpy as np
from common import D, R
from crawl import MODELS
from stats import load
rows = load("val", os.environ.get("SF_VALRUN", "index_rho0.1")); out = {}
for B in (10, 25, 50):
    last = []
    for r in rows.values():
        f = [e for e in r["events"] if e[0] == "F" and e[1] <= B + 1e-9]
        if f: last.append(f[-1][5])
    out[str(B)] = dict(lam=float(np.median(last)), mean=float(np.mean(last)), n=len(last), q25=float(np.percentile(last, 25)), q75=float(np.percentile(last, 75)))
json.dump(out, open(os.path.join(MODELS, "lambda.json"), "w"), indent=1); print(json.dumps(out, indent=1))
