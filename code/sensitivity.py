"""Exploratory sensitivity of the escalation rule to the price, rho and the budget.

Reads only saved calibration arrays and saved confirmatory runs; makes no judge call and writes
only results/exploratory/sensitivity.json.

- lambda grid: recompute the conservative region with Models.tables at lambda * {0.5,0.75,1,1.5,2}
  (rho in {0.1,1}), reporting the number of escalating bins and the share of calibration links in
  them. This is the decision-level rho sensitivity too, because the region depends on lambda*rho.
- budget: truncate the saved B=25 crawls at B in {5,10,15,20,25} and recompute the paired
  TwoCost-minus-scorer contrast. This is a boundary check on the same crawl paths, not a fair
  comparison of B-tuned policies: the rule was priced with lambda(25), so it is not retuned here.
- binning: rebuild the bin edges at 10/20/40 bins from the calibration links with the sampling
  weights saved in posterior.pkl. The weights are the inverse stratified-sampling probabilities,
  so their weighted quantiles recover the reference (pool) distribution; the 20-bin rebuild
  reproduces the deployed edges to within one link, which makes the bin-count comparison
  like-for-like. This is a decision-level check (does any bin pass), not a full policy replay.

  ../.venv-wsl/bin/python sensitivity.py
"""
import glob
import json
import os

import numpy as np

from common import R, load_goals
from crawl import Models, chain_index
from stats import load
from round1_confirm import qid_of, pboot, CONF_LO, CONF_HI

OUT = os.path.join(R, "exploratory", "sensitivity.json")
LAM0 = json.load(open(os.path.join(os.path.dirname(R), "data", "models", "lambda.json")))["25"]["lam"]


def region_stats(T):
    c, reg = T["count"], T["region"]
    top = int(np.argmax(T["gain"]))
    return dict(bins=int(reg.sum()), n_bins=int(len(reg)), esc=float(c[reg].sum() / c.sum()),
                top_bin=top, top_gain=float(T["gain"][top] / T["lam"]),
                top_lcb=float(T["lcb"][top] / T["lam"]))


def lambda_grid(M):
    out = []
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
        for rho in (0.1, 1.0):
            T = M.tables(rho, LAM0 * mult, conservative="all")
            r = region_stats(T)
            r.update(mult=mult, rho=rho, lam=float(LAM0 * mult))
            out.append(r)
    return out


def rho_grid(M):
    out = []
    for rho in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        T = M.tables(rho, LAM0, conservative="all")
        r = region_stats(T)
        r.update(rho=rho, lam=float(LAM0))
        out.append(r)
    return out


def weighted_quantile(x, w, qs):
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    cw = np.cumsum(w); cw /= cw[-1]
    return np.interp(qs, cw, x)


def binning_grid(M):
    """Rebuild edges at 10/20/40 bins from weighted calibration quantiles and re-evaluate the screen.

    The deployed 20-bin edges are quantiles of the full calibration-link pool; the saved weights are
    inverse sampling probabilities, so a weighted quantile of the sampled links recovers that pool.
    """
    c = M.post["cal"]; gp = chain_index(c["prior"], M.beta)[0]; w = np.asarray(c["w"], float)
    orig = M.post["bin_edges"]
    out = []
    try:
        for nb in (10, 20, 40):
            e = np.unique(weighted_quantile(gp, w, np.linspace(0, 1, nb + 1)))
            e[0], e[-1] = 0.0, np.inf
            M.post["bin_edges"] = e
            for rho in (0.1, 1.0):
                T = M.tables(rho, LAM0, conservative="all")
                r = region_stats(T)
                r.update(nbins=int(len(e) - 1), rho=rho, lam=float(LAM0),
                         open=bool(T["region"].any()))
                out.append(r)
    finally:
        M.post["bin_edges"] = orig
    return out


def budget_grid():
    order, seen = [], set()
    for g in load_goals("test"):
        if g["qid"] not in seen:
            seen.add(g["qid"])
            order.append(g["qid"])
    pos = {q: i for i, q in enumerate(order)}
    conf = {q for q, i in pos.items() if CONF_LO <= i < CONF_HI}
    out = {}
    for reg, S in (("trained", ""), ("zeroshot", "_weak")):
        for rho in ("0.1", "1"):
            base = load("test", f"index_r1{S}_rho0.1")
            ours = load("test", f"ours_r1{S}_rho{rho}")
            common = {g for g in base if qid_of(g) in conf and g in ours}
            byq = {}
            for g in common:
                byq.setdefault(qid_of(g), set()).add(base[g]["depth"])
            keep = {q for q, ds in byq.items() if ds == {2, 3, 4}}
            gids = sorted(g for g in common if qid_of(g) in keep)
            q = np.array([qid_of(g) for g in gids])
            rows = {}
            for B in (5, 10, 15, 20, 25):
                v0 = np.array([base[g][f"gold@{B}"] / 2 for g in gids])
                v = np.array([ours[g][f"gold@{B}"] / 2 for g in gids])
                rows[str(B)] = pboot(v - v0, q)
            out[f"{reg}_rho{rho}"] = dict(n=int(len(gids)), nq=int(len(set(q))), by_budget=rows)
    return out


def main():
    M = Models()
    rec = dict(note=("Exploratory. lambda/rho grids recompute the conservative region on the saved "
                     "calibration links; budget truncates the saved B=25 crawl paths and is not a "
                     "fair comparison of B-tuned policies; binning rebuilds weighted quantile edges "
                     "at 10/20/40 bins, and the 20-bin rebuild reproduces the deployed edges."),
               lam0=float(LAM0), lambda_grid=lambda_grid(M), rho_grid=rho_grid(M),
               budget=budget_grid(), binning_grid=binning_grid(M))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
