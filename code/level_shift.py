"""Offline first half of the level-matched recalibration probe (exploratory).

Refits the prior isotonic calibrators and beta on the hard calibration questions only (the test
split is 100% hard), re-estimates the budget price lambda on hard validation crawls, and compares
both with the deployed full-calibration values. Reads only saved rows, runs and the frozen encoder;
makes no judge call and writes nothing outside results/exploratory/.

  ../.venv-wsl/bin/python level_shift.py
"""
import glob
import json
import os
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from common import D, R, load_goals
import lightgbm as lgb
from crawl import K
from stats import load

OUT = os.path.join(R, "exploratory", "level_shift.json")
KIND = "index_B"          # the kind behind the frozen calib.pkl (beta 2.5846579129232894)


def fit_iso(rows, enc):
    X = np.vstack([r["X"][r["in_crawl"]] for r in rows])
    y = np.concatenate([r["y"][r["in_crawl"]] for r in rows])
    cum = np.cumsum(enc.predict(X)[:, :K], 1)
    iso = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(cum[:, k], (y <= k).astype(float))
           for k in range(K)]
    Rc = np.maximum.accumulate(np.clip(np.stack([iso[k].predict(cum[:, k]) for k in range(K)], 1), 1e-5, 1 - 1e-5), 1)
    return dict(iso=iso, X=X, y=y, Rc=Rc, n_links=int(len(y)),
                base_rates=[float((y <= k).mean()) for k in range(K)],
                auc_R1=float(roc_auc_score(y == 0, Rc[:, 0])),
                auc_R3=float(roc_auc_score(y <= 2, Rc[:, 2])))


def beta_of(rows, enc, iso, kind=None):
    ranks = []
    for r in rows:
        X = r["X"][r["in_crawl"]]
        cum = np.cumsum(enc.predict(X)[:, :K], 1)
        R_ = np.maximum.accumulate(np.clip(np.stack([iso[k].predict(cum[:, k]) for k in range(K)], 1), 1e-5, 1 - 1e-5), 1)
        yy = r["y"][r["in_crawl"]]
        par = r["par"][r["in_crawl"]]
        for page, d in zip(r["fetched"], r["fetched_dist"]):
            if d not in (1, 2):
                continue
            sel = np.where(par == page)[0]
            if len(sel) == 0 or not (yy[sel] == d - 1).any():
                continue
            order = sel[np.argsort(-R_[sel, K - 1], kind=kind)] if kind else sel[np.argsort(-R_[sel, K - 1])]
            ranks.append(int(np.where(yy[order] == d - 1)[0][0]) + 1)
    return (float(np.mean(ranks)) if ranks else 1.0), len(ranks), (float(np.median(ranks)) if ranks else None)


def lam_hard():
    levels = {g["qid"]: g["level"] for g in load_goals("val")}
    rows = load("val", "index_rho0.1")
    out = {}
    for tag, keep in (("full", lambda q: True), ("hard", lambda q: levels.get(q) == "hard")):
        last = []
        for r in rows.values():
            if not keep(r.get("qid", "")):
                continue
            f = [e for e in r["events"] if e[0] == "F" and e[1] <= 25 + 1e-9]
            if f:
                last.append(f[-1][5])
        out[tag] = dict(lam=float(np.median(last)), n=len(last),
                        q25=float(np.percentile(last, 25)), q75=float(np.percentile(last, 75))) if last else None
    return out


def main():
    rows_all = []
    for f in sorted(glob.glob(os.path.join(D, "rows", f"cal_{KIND}_s*.pkl"))):
        rows_all += pickle.load(open(f, "rb"))
    levels = {g["qid"]: g["level"] for g in load_goals("cal")}
    qid = lambda r: r["gid"].rsplit("_h", 1)[0]
    rows_h = [r for r in rows_all if levels.get(qid(r)) == "hard"]
    enc = lgb.Booster(model_file=os.path.join(D, "models", "encoder.txt"))

    full = fit_iso(rows_all, enc)
    hard = fit_iso(rows_h, enc)
    beta_full, npg_full, med_full = beta_of(rows_all, enc, full["iso"])
    beta_hard, npg_hard, med_hard = beta_of(rows_h, enc, hard["iso"])
    beta_full_s = beta_of(rows_all, enc, full["iso"], "stable")[0]
    beta_hard_s = beta_of(rows_h, enc, hard["iso"], "stable")[0]
    lam = lam_hard()

    rec = dict(kind=KIND,
               note=("Exploratory. The full-row refit on WSL gives beta 2.5473/2.5660 depending on the "
                     "argsort tie order, not the frozen 2.5847; the isotonic curves and rows match, so the "
                     "gap is tie-breaking in the rank statistic, not data drift. Saved runs and paper "
                     "numbers use the frozen beta and are unaffected. The level comparison uses the refit "
                     "full as its baseline; run the same fit on the original platform before quoting it."),
               n_questions=dict(full=len(levels), hard=sum(1 for v in levels.values() if v == "hard")),
               n_rows=dict(full=len(rows_all), hard=len(rows_h)),
               n_links=dict(full=full["n_links"], hard=hard["n_links"]),
               base_rates=dict(full=full["base_rates"], hard=hard["base_rates"]),
               auc_R1=dict(full=full["auc_R1"], hard=hard["auc_R1"]),
               auc_R3=dict(full=full["auc_R3"], hard=hard["auc_R3"]),
               beta=dict(full=beta_full, hard=beta_hard, full_stable=beta_full_s, hard_stable=beta_hard_s,
                         deployed=2.5846579129232894),
               n_pages_for_beta=dict(full=npg_full, hard=npg_hard),
               median_rank=dict(full=med_full, hard=med_hard),
               lambda_hard=lam)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
