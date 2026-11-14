"""Proposition-1 gain curve on the confirmatory test blocks, using only cached judge calls.

The confirmatory crawls never judged most test links, so a full replay is impossible without new
calls.  This script uses every confirmatory test link for which a judge signal is already cached
(qids at positions 80--199 of the test file) and reports the realised gain of acting on the
posterior instead of the prior, by deployed prior-index bin, with a cluster bootstrap over
questions.  Coverage is reported per bin and per question: the covered links are the links that
exploratory policies happened to inspect, so the curve is a selected subsample, not a replay.

No judge call, no run file, and no model is modified.  Writes
results/exploratory/prop1_confirm_cached.json.
"""
import glob
import json
import os
import pickle
import sqlite3

import numpy as np

from common import D, R, load_goals
from crawl import chain_index, K, realised_payoff

OUT = os.path.join(R, "exploratory", "prop1_confirm_cached.json")
CONF_LO, CONF_HI = 80, 200
B = 25
NBOOT = 4000


def load_cache():
    db = sqlite3.connect(os.path.join(D, "cache", "llm.sqlite"))
    rows = db.execute("SELECT qid,parent,target,probs FROM insp WHERE model='qwen2.5:7b'").fetchall()
    out = {}
    for qid, par, tgt, pj in rows:
        p = np.array(json.loads(pj))
        out[(qid, int(par), int(tgt))] = (float((p * np.arange(10)).sum() / 9.0), float(p[5:].sum()))
    return out


def boot_bin(g, qid):
    """Mean and 95% interval of the bin mean, resampling questions."""
    uq, inv = np.unique(qid, return_inverse=True)
    s = np.bincount(inv, g, len(uq))
    c = np.bincount(inv, minlength=len(uq)).astype(float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(uq), (NBOOT, len(uq)))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s[idx].sum(1) / c[idx].sum(1)
    m = m[np.isfinite(m)]
    return (float(g.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            int(len(uq))) if len(m) else (float(g.mean()), float("nan"), float("nan"), int(len(uq)))


def main():
    order, seen = [], set()
    for go in load_goals("test"):
        if go["qid"] not in seen:
            seen.add(go["qid"])
            order.append(go["qid"])
    pos = {q: i for i, q in enumerate(order)}
    conf_q = {q for q, i in pos.items() if CONF_LO <= i < CONF_HI}
    cache = load_cache()
    print("cache entries", len(cache), "confirmatory qids", len(conf_q), flush=True)

    out = {"config": {"blocks": [80, 120, 160], "questions": len(conf_q), "budget": B,
                      "note": "cached-signal subsample; selected, not a replay"},
           "cells": {}}
    for tag, kind, mod in (("trained", "index_T", "models"), ("zeroshot", "index_weakT", "models_weak")):
        post = pickle.load(open(os.path.join(D, mod, "posterior.pkl"), "rb"))
        edges = np.asarray(post["bin_edges"], float)
        nb = len(edges) - 1
        lam = json.load(open(os.path.join(D, mod, "lambda.json")))["25"]["lam"]
        beta = pickle.load(open(os.path.join(D, mod, "calib.pkl"), "rb"))["beta"]
        rows = []
        for f in sorted(glob.glob(os.path.join(D, "rows", f"test_{kind}_s*.pkl"))):
            rows += pickle.load(open(f, "rb"))
        recs = []
        for r in rows:
            qid = r["gid"].rsplit("_h", 1)[0]
            if qid not in conf_q:
                continue
            m = r["in_crawl"]
            par, tgt = r["par"][m], r["tgt"][m]
            pr, yy = r["prior"][m], r["y"][m]
            gp = chain_index(pr, beta)[0]
            b = np.clip(np.searchsorted(edges, gp, side="right") - 1, 0, nb - 1)
            for j in range(len(gp)):
                key = (qid, int(par[j]), int(tgt[j]))
                if key not in cache:
                    continue
                s, phi = cache[key]
                recs.append((qid, int(b[j]), pr[j].copy(), int(yy[j]), s, phi))
        print(tag, "confirmatory in-crawl links with a cached call", len(recs), flush=True)
        if not recs:
            continue
        qid = np.array([x[0] for x in recs])
        b = np.array([x[1] for x in recs])
        pr = np.stack([x[2] for x in recs])
        yy = np.array([x[3] for x in recs])
        s = np.array([x[4] for x in recs])
        phi = np.array([x[5] for x in recs])
        pq = post["model"].predict(pr, s, phi)
        g = realised_payoff(pq, yy, lam, beta)[0] - realised_payoff(pr, yy, lam, beta)[0]
        # all confirmatory in-crawl links, for coverage
        allb = []
        for r in rows:
            q2 = r["gid"].rsplit("_h", 1)[0]
            if q2 not in conf_q:
                continue
            m = r["in_crawl"]
            gp = chain_index(r["prior"][m], beta)[0]
            allb += list(np.clip(np.searchsorted(edges, gp, side="right") - 1, 0, nb - 1))
        allb = np.array(allb)
        bins = []
        for j in range(nb):
            m = b == j
            n_all = int((allb == j).sum())
            n_cov = int(m.sum())
            if n_cov == 0:
                continue
            mean, lo, hi, nq = boot_bin(g[m], qid[m])
            bins.append(dict(bin=j, n_cov=n_cov, n_all=n_all,
                             coverage=(n_cov / max(n_all, 1)),
                             mean_gain=mean, lo95=lo, hi95=hi, nq=nq,
                             passes=bool(lo > lam * 0.1)))
        overall = boot_bin(g, qid)
        out["cells"][tag] = {"lam": lam, "beta": beta, "n_cached": int(len(recs)),
                             "coverage": float(len(recs) / max(len(allb), 1)),
                             "overall": dict(mean=overall[0], lo95=overall[1], hi95=overall[2]),
                             "bins": bins}
        print(f"  overall gain {overall[0]:+.6f} [{overall[1]:+.6f},{overall[2]:+.6f}] "
              f"coverage {len(recs)/max(len(allb),1):.3f}", flush=True)
        for x in bins:
            print(f"  bin {x['bin']:2d} cov {x['coverage']:.2f} n={x['n_cov']:5d} "
                  f"gain {x['mean_gain']:+.6f} [{x['lo95']:+.6f},{x['hi95']:+.6f}]", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
