"""Calibration-only judge-worthiness screen, swept over scorer strength.

The paper's rule needs a judge-worthiness decision before any test-time call: is there any prior
index bin in which a call pays for itself?  The screen is the deployed conservative test, computed
on calibration links only: the screen is positive iff at least one bin's one-sided 95% lower bound
on realised gain exceeds lambda*rho.  This script asks whether that screen predicts the realised
value of judging on questions the screen did not see, as the cheap scorer is weakened.

Scorer strengths (nested LightGBM heads, same features and parameters as the deployed head):
  deployed   the frozen encoder (anchor to the paper)
  n771       retrained on all 771 train questions
  n200       retrained on 200 train questions
  n50        retrained on 50 train questions
  dense      the zero-shot bge similarity, no trained head
For each strength we refit the isotonic prior, beta and the cross-fitted offset posterior on the
same cached calibration links and the same cached judge signals.  No new judge call is made.

Protocol (declared before the run)
----------------------------------
* Five folds over the 400 calibration questions (seed 20261113).  For each fold, the 20-bin table,
  its 95% one-sided LCB and the screen are built on the other four folds only, then applied to the
  held-out fold.  Every question is screened by a table that never saw it.
* screen_max = max_b(lcb_b - lambda*rho); screen_positive = any region bin.
* Outcomes on held-out links: mean net Lagrangian gain per link of the screen's rule, with a
  cluster bootstrap over held-out questions (4,000 resamples).  Reported on the hard-question
  population (the deployment population) and on all held-out links (the calibration population).
* Also scored with the same tuning budget: an entropy router that judges the links with the largest
  layer-distribution entropy, threshold tuned on the four training folds to maximise training net.
* The screen is supported only if, across folds, negative-screen folds do not show a positive net
  gain and positive-screen folds do.  Otherwise the result is reported as a failure of the screen.

Writes results/exploratory/scorer_strength_screen.json.
"""
import glob
import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from common import D, R, FEATS
from crawl import K, chain_index, realised_payoff
from twocost_ablation import (load_ctx, raw_cum, fit_iso, apply_iso, beta_of_rows,
                              edges_from, build_table, fit_offset)

OUT = os.path.join(R, "exploratory", "scorer_strength_screen.json")
SEED_SPLIT = 20261113
SEED_SUBSET = 0
RHOS = (0.1, 1.0)
STRENGTHS = ("deployed", "n771", "n200", "n50", "dense")
NBINS = 20
CONF = 0.95
NBOOT_TABLE = 1000
NBOOT_CI = 4000
HARD_LEVEL = "hard"
NFOLD = 5


def load_train_rows():
    rows = []
    for kind in ("behav", "index"):
        for f in sorted(glob.glob(os.path.join(D, "rows", f"train_{kind}_s*.pkl"))):
            rows += pickle.load(open(f, "rb"))
    return rows


def train_head(rows, strength):
    qids = sorted({r["gid"].rsplit("_h", 1)[0] for r in rows})
    perm = np.random.default_rng(SEED_SUBSET).permutation(qids)
    n = len(qids) if strength == "n771" else int(strength[1:])
    subset = list(perm[:n])
    nval = max(1, n // 10)
    va_q = set(subset[-nval:])
    tr_q = set(subset[:-nval])
    tr = [r for r in rows if r["gid"].rsplit("_h", 1)[0] in tr_q]
    va = [r for r in rows if r["gid"].rsplit("_h", 1)[0] in va_q]
    Xt = np.vstack([r["X"] for r in tr])
    yt = np.concatenate([r["y"] for r in tr])
    Xv = np.vstack([r["X"] for r in va])
    yv = np.concatenate([r["y"] for r in va])
    params = dict(objective="multiclass", num_class=K + 1, learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, seed=0)
    bst = lgb.train(params, lgb.Dataset(Xt, yt, feature_name=FEATS), 400,
                    valid_sets=[lgb.Dataset(Xv, yv)],
                    callbacks=[lgb.early_stopping(40, verbose=False)])
    return bst, dict(n_questions=n, n_train_links=int(len(yt)), best_iter=int(bst.best_iteration))


def layer_entropy(Rc):
    p = np.column_stack([Rc[:, 0], Rc[:, 1] - Rc[:, 0], Rc[:, 2] - Rc[:, 1], 1 - Rc[:, 2]])
    p = np.clip(p, 1e-9, 1)
    return -(p * np.log(p)).sum(1)


def net_vector(prior, post, y, beta, T, lam, rho, mask):
    gp = chain_index(prior, beta)[0]
    b = np.clip(np.searchsorted(T["edges"], gp, side="right") - 1, 0, T["nb"] - 1)
    reg = T["region"][b] & mask
    Ruse = np.where(reg[:, None], post, prior)
    pay_use, _ = realised_payoff(Ruse, y, lam, beta)
    pay_prior, _ = realised_payoff(prior, y, lam, beta)
    return (pay_use - pay_prior) - lam * rho * reg.astype(float), reg


def per_q(v, q, mask):
    uq, inv = np.unique(q[mask], return_inverse=True)
    s = np.bincount(inv, v[mask], len(uq))
    c = np.bincount(inv, minlength=len(uq)).astype(float)
    return {str(a): float(x / y) for a, x, y in zip(uq, s, c)}


def wmean(v, w):
    return float((v * w).sum() / max(w.sum(), 1e-12))


def boot_mean(d, nboot=NBOOT_CI, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (nboot, len(d)))
    m = d[idx].mean(1)
    return dict(mean=float(d.mean()), lo95=float(np.percentile(m, 2.5)),
                hi95=float(np.percentile(m, 97.5)),
                p=float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))), n=int(len(d)))


def entropy_router(prior, post, y, qid, beta, lam, rho, tr, te):
    H = layer_entropy(prior)
    best = None
    for qq in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        thr = float(np.quantile(H[tr], qq)) if qq > 0 else np.inf
        sel = H >= thr
        Ruse = np.where((sel & tr)[:, None], post, prior)
        pay_u, _ = realised_payoff(Ruse, y, lam, beta)
        pay_p, _ = realised_payoff(prior, y, lam, beta)
        net = (pay_u - pay_p) - lam * rho * (sel & tr)
        score = float(net[tr].mean())
        if best is None or score > best[0]:
            best = (score, thr, qq)
    _, thr, qq = best
    sel = H >= thr
    Ruse = np.where(sel[:, None], post, prior)
    pay_u, _ = realised_payoff(Ruse, y, lam, beta)
    pay_p, _ = realised_payoff(prior, y, lam, beta)
    net = (pay_u - pay_p) - lam * rho * sel
    return dict(quantile=qq, threshold=thr, calls=float(sel[te].mean()),
                net=float(net[te].mean()), per_q=per_q(net, qid, te))


def main():
    print("loading trained context", flush=True)
    ctx = load_ctx("trained")
    qid, y, w = ctx["qid"], ctx["y"], ctx["w"]
    X = ctx["X"]
    lam = ctx["lam"]
    uq = np.unique(qid)
    perm = np.random.default_rng(SEED_SPLIT).permutation(uq)
    fold = {q: i % NFOLD for i, q in enumerate(perm)}
    fo_q = np.array([fold[q] for q in qid])
    hard = np.array([ctx["levels"][q] == HARD_LEVEL for q in qid])
    allmask = np.ones(len(qid), bool)
    print(f"nfolds={NFOLD} hard links={hard.sum()} all links={len(qid)}", flush=True)

    cal_rows = ctx["rows"]
    Xcal = np.vstack([r["X"][r["in_crawl"]] for r in cal_rows])
    ycal = np.concatenate([r["y"][r["in_crawl"]] for r in cal_rows])
    train_rows = load_train_rows()

    # test rows, shared by all strengths
    trows = []
    for f in sorted(glob.glob(os.path.join(D, "rows", "test_index_T_s*.pkl"))):
        trows += pickle.load(open(f, "rb"))
    Xt = np.vstack([r["X"][r["in_crawl"]] for r in trows])
    yt = np.concatenate([r["y"][r["in_crawl"]] for r in trows])

    out = {"protocol": {
        "folds": NFOLD, "split_seed": SEED_SPLIT, "subset_seed": SEED_SUBSET,
        "screen": "positive iff max_b(lcb_b - lambda*rho) > 0 with 95% one-sided LCB, 20 prior-index bins, built on 4 folds",
        "outcome": "mean net Lagrangian gain per held-out link; cluster bootstrap over held-out questions",
        "populations": ["hard", "all"],
        "router": "entropy router tuned on the training folds to maximise training net",
        "lambda": lam, "rho": list(RHOS), "strengths": list(STRENGTHS),
        "no_new_judge_calls": True,
        "caveat": "offline calibration-link screen; deployment confirmation needs test crawls"},
        "cells": {}}

    for strength in STRENGTHS:
        print("=== strength", strength, flush=True)
        rec = {"strength": strength}
        if strength == "deployed":
            bst = lgb.Booster(model_file=os.path.join(D, "models", "encoder.txt"))
            dense = False
            rec["train"] = {"n_questions": 771, "note": "frozen deployed encoder"}
        elif strength == "dense":
            bst, dense = None, True
            rec["train"] = {"n_questions": 0, "note": "zero-shot dense similarity"}
        else:
            bst, info = train_head(train_rows, strength)
            dense = False
            rec["train"] = info
        cum_cal = raw_cum(Xcal, bst, dense)
        iso = fit_iso(cum_cal, ycal)
        prior = apply_iso(iso, raw_cum(X, bst, dense))
        beta, npages = beta_of_rows(cal_rows, iso, bst, dense)
        rec["beta"], rec["beta_pages"] = beta, int(npages)
        post = np.zeros_like(prior)
        for f in range(NFOLD):
            a, b = fo_q != f, fo_q == f
            post[b] = fit_offset(prior, ctx["s"], ctx["phi"], y, a).predict(
                prior[b], ctx["s"][b], ctx["phi"][b])
        rec["auc_cal_r1"] = float(roc_auc_score((y == 0).astype(int), prior[:, 0]))
        pt = apply_iso(iso, raw_cum(Xt, bst, dense))
        rec["auc_test_r1"] = float(roc_auc_score((yt == 0).astype(int), pt[:, 0]))
        rec["auc_test_r3"] = float(roc_auc_score((yt <= 2).astype(int), pt[:, 2]))
        print(f"  beta={beta:.3f} auc_cal={rec['auc_cal_r1']:.4f} "
              f"auc_test={rec['auc_test_r1']:.4f}/{rec['auc_test_r3']:.4f}", flush=True)

        for rho in RHOS:
            folds = []
            for f in range(NFOLD):
                tr, te = fo_q != f, fo_q == f
                T = build_table(prior[tr], post[tr], y[tr], w[tr], qid[tr], beta,
                                edges_from(prior[tr], NBINS), lam, rho, conf=CONF,
                                region_mode="all", nboot=NBOOT_TABLE)
                cell = {"fold": f, "screen_positive": bool(T["region"].any()),
                        "screen_max": float(np.max(T["lcb"] - lam * rho)),
                        "region_bins": int(T["region"].sum()),
                        "region_share": float(T["count"][T["region"]].sum() / max(T["count"].sum(), 1)),
                        "gain": [float(v) for v in T["gain"]],
                        "lcb": [float(v) for v in T["lcb"]]}
                for pop, mask in (("hard", hard), ("all", allmask)):
                    net, sel = net_vector(prior, post, y, beta, T, lam, rho, te & mask)
                    cell[pop] = {"calls": float(sel[te & mask].mean()),
                                 "net_unweighted": float(net[te & mask].mean()),
                                 "net_weighted": wmean(net[te & mask], w[te & mask]),
                                 "per_q": per_q(net, qid, te & mask)}
                cell["router"] = entropy_router(prior, post, y, qid, beta, lam, rho, tr, te & hard)
                folds.append(cell)
            rec[f"rho{rho}"] = {"folds": folds}
            for pop in ("hard", "all"):
                pooled = {}
                for c in folds:
                    pooled.update(c[pop]["per_q"])
                npos = sum(c["screen_positive"] for c in folds)
                net_pos = [c[pop]["net_unweighted"] for c in folds if c["screen_positive"]]
                net_neg = [c[pop]["net_unweighted"] for c in folds if not c["screen_positive"]]
                rec[f"rho{rho}"][pop] = {
                    "screen_positive_folds": int(npos),
                    "calls": float(np.mean([c[pop]["calls"] for c in folds])),
                    "net_unweighted": float(np.mean([c[pop]["net_unweighted"] for c in folds])),
                    "net_weighted": float(np.mean([c[pop]["net_weighted"] for c in folds])),
                    "boot": boot_mean(list(pooled.values())),
                    "net_when_screen_pos": float(np.mean(net_pos)) if net_pos else None,
                    "net_when_screen_neg": float(np.mean(net_neg)) if net_neg else None,
                }
            rp = rec[f"rho{rho}"]
            print(f"  rho={rho} screen + in {rp['hard']['screen_positive_folds']}/{NFOLD} folds  "
                  f"hard calls={rp['hard']['calls']:.3f} net={rp['hard']['net_unweighted']:+.6f} "
                  f"[{rp['hard']['boot']['lo95']:+.6f},{rp['hard']['boot']['hi95']:+.6f}]  "
                  f"all net={rp['all']['net_unweighted']:+.6f} "
                  f"[{rp['all']['boot']['lo95']:+.6f},{rp['all']['boot']['hi95']:+.6f}]", flush=True)
            router = {k: (float(np.mean([c["router"][k] for c in folds])) if k != "per_q" else None)
                      for k in ("calls", "net")}
            rec[f"rho{rho}"]["router"] = router
        out["cells"][strength] = rec

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)

    print("\nstrength  rho  screen+  calls   hard_net     95% CI              all_net      95% CI             auc")
    for s in STRENGTHS:
        r = out["cells"][s]
        for rho in RHOS:
            h = r[f"rho{rho}"]["hard"]
            a = r[f"rho{rho}"]["all"]
            print(f"{s:9s} {rho:3.1f}  {h['screen_positive_folds']}/{NFOLD}    {h['calls']:.3f}  "
                  f"{h['net_unweighted']:+.6f} [{h['boot']['lo95']:+.6f},{h['boot']['hi95']:+.6f}]  "
                  f"{a['net_unweighted']:+.6f} [{a['boot']['lo95']:+.6f},{a['boot']['hi95']:+.6f}]  "
                  f"{r['auc_test_r1']:.3f}")


if __name__ == "__main__":
    main()
