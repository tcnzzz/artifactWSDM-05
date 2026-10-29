"""Confirmatory analysis restricted to the completed test blocks (questions 80..199).

This replaces the partially-run round1_analyze.py for the confirmatory table. It reads only
saved run files under results/runs/test and writes results/tables/round1_confirm.json. It does
not modify any saved run, log, table or manuscript file.

Scope decisions (author, 2026-10-29):
- The confirmatory set is the three completed blocks, test-question positions 80..199 (n=120
  questions, 3 depths each -> 360 crawls per cell). Block 200-240 was killed mid-run (two
  cells complete for the trained scorer at rho=0.1 plus two partial band records) and blocks
  240-399 were never started; none of them enters the confirmatory analysis.
- Questions without all three constructed depths in every cell are dropped from paired
  contrasts (drops the two partial band-cascade questions 5adfa6db..., 5a860384...).
- Analysis as preregistered in preregistration/ROUND1_PLAN.md (amendment 2): paired cluster
  bootstrap over questions, 10,000 resamples, seed 0; Holm within the family vs scorer alone
  (4 contrasts) and within TwoCost vs baselines (3 contrasts); TOST at margin 0.02.
"""
import glob
import json
import os

import numpy as np

from common import R, load_goals
from stats import load

OUT = os.path.join(R, "tables", "round1_confirm.json")
B = 25
DELTA = 0.02
NBOOT = 10000
RHOS = ["0.1", "1"]
REGS = (("trained", ""), ("zeroshot", "_weak"))
POLS = ("ours", "topm", "thompson", "band")
CONF_LO, CONF_HI = 80, 200


def qid_of(gid):
    return gid.rsplit("_h", 1)[0]


def vec(rows, gids, key="gold"):
    return np.array([rows[g][f"{key}@{B}"] / (2.0 if key == "gold" else 1.0) for g in gids])


def pboot(d, q, seed=0):
    """Paired cluster bootstrap over questions.

    Returns mean, 95% and 90% percentile intervals, the two-sided superiority p-value, the two
    one-sided TOST p-values at DELTA and the equivalence verdict (90% interval inside (-D, D)).
    """
    d = np.asarray(d, float)
    uq, inv = np.unique(q, return_inverse=True)
    rng = np.random.default_rng(seed)
    s = np.bincount(inv, d, len(uq))
    c = np.bincount(inv, minlength=len(uq)).astype(float)
    idx = rng.integers(0, len(uq), (NBOOT, len(uq)))
    m = s[idx].sum(1) / c[idx].sum(1)
    p_sup = float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
    p_lo = float((m <= -DELTA).mean())   # H0: effect <= -DELTA  (reject => non-inferior above -DELTA)
    p_hi = float((m >= DELTA).mean())    # H0: effect >= +DELTA
    lo95, hi95 = np.percentile(m, [2.5, 97.5])
    lo90, hi90 = np.percentile(m, [5, 95])
    if lo90 > -DELTA and hi90 < DELTA:
        verdict = "equiv"
    elif hi95 < -DELTA:
        verdict = "below"
    elif lo95 > DELTA:
        verdict = "above"
    else:
        verdict = "inconclusive"
    return dict(mean=float(d.mean()), lo95=float(lo95), hi95=float(hi95),
                lo90=float(lo90), hi90=float(hi90), p=p_sup,
                p_tost_lower=p_lo, p_tost_upper=p_hi, p_tost=float(max(p_lo, p_hi)),
                p_noninf=float(p_lo), verdict=verdict,
                equiv=bool(verdict == "equiv"), n=int(len(d)), nq=int(len(uq)))


def holm(ps):
    o = np.argsort(ps)
    adj = np.zeros(len(ps))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


def main():
    order = []
    seen = set()
    for g in load_goals("test"):
        if g["qid"] not in seen:
            seen.add(g["qid"])
            order.append(g["qid"])
    pos = {q: i for i, q in enumerate(order)}
    conf_q = {q for q, i in pos.items() if CONF_LO <= i < CONF_HI}

    runs = {}
    for d in glob.glob(os.path.join(R, "runs", "test", "*_r1*")):
        n = os.path.basename(d)
        rows = {g: r for g, r in load("test", n).items()
                if qid_of(g) in conf_q and 2 <= r["depth"] <= 4}
        if rows:
            runs[n] = rows

    out = {"config": {"blocks": [80, 120, 160], "questions": len(conf_q), "budget": B,
                      "delta_tost": DELTA, "nboot": NBOOT, "seed": 0,
                      "excluded_partial_qids": ["5adfa6db55429942ec259ad6", "5a86038455429960ec39b600"],
                      "excluded_reason": "block 200-240 killed mid-run; one of three depths recorded"},
          "cells": {}}

    for reg, S in REGS:
        base = f"index_r1{S}_rho0.1"
        if base not in runs:
            continue
        out["cells"][reg] = {}
        for rho in RHOS:
            names = {p: f"{p}_r1{S}_rho{rho}" for p in POLS if f"{p}_r1{S}_rho{rho}" in runs}
            # intersection of crawl ids present in every cell, and only questions with all 3 depths
            common = set(runs[base])
            for n in names.values():
                common &= set(runs[n])
            byq = {}
            for g in common:
                byq.setdefault(qid_of(g), set()).add(runs[base][g]["depth"])
            keep_q = {q for q, ds in byq.items() if ds == {2, 3, 4}}
            gids = sorted(g for g in common if qid_of(g) in keep_q)
            q = np.array([qid_of(g) for g in gids])
            dep = np.array([runs[base][g]["depth"] for g in gids])
            v0 = vec(runs[base], gids)
            cell = {"n": len(gids), "nq": int(len(set(q))), "rho": float(rho),
                    "scorer": {"recall": float(v0.mean()), "calls": 0.0}}
            for p, n in names.items():
                v = vec(runs[n], gids)
                calls = vec(runs[n], gids, "llm")
                cost = vec(runs[n], gids, "fetch") + float(rho) * calls
                cell[p] = {"recall": float(v.mean()), "calls": float(calls.mean()),
                           "max_cost": float(cost.max()),
                           "diff_crawls": int(np.sum(v != v0)),
                           "vs_scorer": pboot(v - v0, q),
                           "by_depth": {str(k): pboot((v - v0)[dep == k], q[dep == k]) for k in (2, 3, 4)}}
                if p != "ours" and "ours" in names:
                    cell[p]["ours_minus"] = pboot(vec(runs[names["ours"]], gids) - v, q)
                    cell[p]["ours_minus"]["ours_calls"] = float(vec(runs[names["ours"]], gids, "llm").mean())
            fam = [p for p in POLS if p in cell]
            adj = holm([cell[p]["vs_scorer"]["p"] for p in fam])
            for p, a in zip(fam, adj):
                cell[p]["vs_scorer"]["p_holm"] = float(a)
            cell["ours"]["vs_scorer_holm_only"] = cell["ours"]["vs_scorer"]["p_holm"]
            fam2 = [p for p in ("topm", "thompson", "band") if p in cell and "ours_minus" in cell[p]]
            adj2 = holm([cell[p]["ours_minus"]["p"] for p in fam2])
            for p, a in zip(fam2, adj2):
                cell[p]["ours_minus"]["p_holm"] = float(a)
            out["cells"][reg][rho] = cell

    # Prediction 2 (judge-free) on the confirmatory questions, and the wider 320-question
    # version for comparison, using the same saved runs.
    idx_conf = runs.get("index_r1_rho0.1")
    idx_all = load("test", "index_rho0.1")
    out["prediction2"] = {}
    if idx_conf:
        out["prediction2"]["determinism_identical_frac"] = float(np.mean(
            [idx_conf[g]["gold@25"] == idx_all[g]["gold@25"] for g in idx_conf]))
    for tag, idx_rows, lo, hi in (("confirm_80_200", idx_conf, CONF_LO, CONF_HI),
                                  ("wide_80_400", idx_all, 80, 400)):
        if not idx_rows:
            continue
        qset = {qq for qq, i in pos.items() if lo <= i < hi}
        sub = {}
        for my in ("greedy", "greedy_bin"):
            rm = load("test", f"{my}_rho0.1")
            gids = sorted(g for g in set(idx_rows) & set(rm) if qid_of(g) in qset)
            q = np.array([qid_of(g) for g in gids])
            dep = np.array([idx_rows[g]["depth"] for g in gids])
            d = vec(idx_rows, gids) - vec(rm, gids)
            sub[my] = {str(k): pboot(d[dep == k], q[dep == k]) for k in (2, 3, 4)}
            sub[my]["all"] = pboot(d, q)
        out["prediction2"][tag] = sub

    lv = {}
    for sp in ("train", "cal", "val", "test"):
        gq = {g["qid"]: g["level"] for g in load_goals(sp)}
        lv[sp] = {l: sum(1 for x in gq.values() if x == l) for l in ("easy", "medium", "hard")}
    out["levels"] = lv

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    for reg, cells in out["cells"].items():
        for rho, c in cells.items():
            print(f"\n{reg} rho={rho} n={c['n']} nq={c['nq']} scorer={c['scorer']['recall']:.6f}")
            for p in POLS:
                if p in c:
                    v = c[p]["vs_scorer"]
                    print(f"  {p:9s} recall={c[p]['recall']:.6f} calls={c[p]['calls']:.3f} "
                          f"d={v['mean']:+.4f} [{v['lo95']:+.4f},{v['hi95']:+.4f}] "
                          f"p={v['p']:.4f} pHolm={v['p_holm']:.4f} TOST={v['verdict']:12s} pTOST={v['p_tost']:.4f}")
                    if "ours_minus" in c[p]:
                        o = c[p]["ours_minus"]
                        print(f"    ours_minus_{p}: {o['mean']:+.4f} [{o['lo95']:+.4f},{o['hi95']:+.4f}] "
                              f"p={o['p']:.4f} pHolm={o['p_holm']:.4f} TOST={o['verdict']}")
    print("\nprediction2:")
    print(json.dumps(out["prediction2"], indent=1)[:2500])


if __name__ == "__main__":
    main()
