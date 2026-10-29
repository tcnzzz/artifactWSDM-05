"""Calibrate the encoder on calibration goals (disjoint from the goals the encoder was trained on) and estimate beta.
Isotonic regression per cumulative layer probability, fitted on links met by the index policy on calibration goals.
beta = mean rank of the first progress link on a page that is 1 or 2 hops from gold (expected fetches to descend one level).
  python fit_calib.py <rows-kind>        e.g. index_A  (rows file data/rows/cal_<kind>_s*.pkl)
"""
import sys, os, glob, pickle, json
import numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from common import D
from crawl import MODELS, K, chain_index
kind = sys.argv[1]; dense = "--dense" in sys.argv; rows = []
for f in sorted(glob.glob(os.path.join(D, "rows", f"cal_{kind}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
X = np.vstack([r["X"][r["in_crawl"]] for r in rows]); y = np.concatenate([r["y"][r["in_crawl"]] for r in rows])
if dense: cum = np.tile((0.5 * (X[:, 0] + X[:, 1]))[:, None], (1, K))        # weak regime: zero-shot dense similarity, no trained head
else: enc = lgb.Booster(model_file=os.path.join(MODELS, "encoder.txt")); cum = np.cumsum(enc.predict(X)[:, :K], 1)
iso = []
for k in range(K):
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(cum[:, k], (y <= k).astype(float)); iso.append(ir)
Rc = np.maximum.accumulate(np.clip(np.stack([iso[k].predict(cum[:, k]) for k in range(K)], 1), 1e-5, 1 - 1e-5), 1)
ranks = []; off = 0
for r in rows:
    m = r["in_crawl"]; n = int(m.sum()); R_ = Rc[off:off + n]; par = r["par"][m]; yy = r["y"][m]; off += n
    for page, d in zip(r["fetched"], r["fetched_dist"]):
        if d not in (1, 2): continue
        sel = np.where(par == page)[0]
        if len(sel) == 0 or not (yy[sel] == d - 1).any(): continue
        order = sel[np.argsort(-R_[sel, K - 1])]; ranks.append(int(np.where(yy[order] == d - 1)[0][0]) + 1)
beta = float(np.mean(ranks)) if ranks else 1.0
pickle.dump(dict(iso_prior=iso, beta=beta, dense=dense), open(os.path.join(MODELS, "calib.pkl"), "wb"))
from sklearn.metrics import roc_auc_score
info = dict(kind=kind, dense=dense, auc_R1=float(roc_auc_score(y == 0, Rc[:, 0])), auc_R3=float(roc_auc_score(y <= 2, Rc[:, 2])), n_links=int(len(y)), base_rates=[float((y <= k).mean()) for k in range(K)], beta=beta, n_pages_for_beta=len(ranks), median_rank=float(np.median(ranks)) if ranks else None)
json.dump(info, open(os.path.join(MODELS, f"calib_info_{kind}.json"), "w"), indent=1); print(json.dumps(info, indent=1))
