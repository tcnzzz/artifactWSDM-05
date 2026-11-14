"""Minimum detectable effect of the confirmatory design (120 questions, three depths each).

Reads only saved confirmatory run files; makes no judge call.  For each judge policy it forms the
per-question mean recall difference to the scorer alone, takes the analytic cluster standard error
(question is the independent unit), and reports the two-sided 80%-power MDE:

    MDE = (z_{0.975} + z_{0.80}) * SE = 2.8016 * SE.

The point is to state what "TOST-equivalent" and "inconclusive" mean at this sample size, and
whether the 0.02 equivalence margin is resolvable.

Write results/exploratory/mde.json.
"""
import glob
import json
import os

import numpy as np

from common import R, load_goals
from stats import load
from round1_confirm import qid_of, CONF_LO, CONF_HI

OUT = os.path.join(R, "exploratory", "mde.json")
B = 25
Z = 1.959964 + 0.841621  # two-sided alpha=0.05, power=0.80


def per_question(rows, gids, key="gold"):
    v = np.array([rows[g][f"{key}@{B}"] / (2.0 if key == "gold" else 1.0) for g in gids])
    q = np.array([qid_of(g) for g in gids])
    return np.array([v[q == u].mean() for u in np.unique(q)])


def main():
    order, seen = [], set()
    for g in load_goals("test"):
        if g["qid"] not in seen:
            seen.add(g["qid"])
            order.append(g["qid"])
    pos = {q: i for i, q in enumerate(order)}
    conf = {q for q, i in pos.items() if CONF_LO <= i < CONF_HI}

    runs = {}
    for d in glob.glob(os.path.join(R, "runs", "test", "*_r1*")):
        n = os.path.basename(d)
        rows = {g: r for g, r in load("test", n).items()
                if qid_of(g) in conf and 2 <= r["depth"] <= 4}
        if rows:
            runs[n] = rows

    out = {"config": {"questions": len(conf), "budget": B, "alpha": 0.05, "power": 0.80,
                      "z_sum": Z, "unit": "question", "depths": [2, 3, 4]}, "cells": {}}
    for reg, S in (("trained", ""), ("zeroshot", "_weak")):
        base = f"index_r1{S}_rho0.1"
        for pol in ("ours", "topm", "thompson", "band"):
            nm = f"{pol}_r1{S}_rho0.1"
            if base not in runs or nm not in runs:
                continue
            common = set(runs[base]) & set(runs[nm])
            byq = {}
            for g in common:
                byq.setdefault(qid_of(g), set()).add(runs[base][g]["depth"])
            keep = {q for q, ds in byq.items() if ds == {2, 3, 4}}
            gids = sorted(g for g in common if qid_of(g) in keep)
            d = per_question(runs[nm], gids) - per_question(runs[base], gids)
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            out["cells"][f"{reg}_{pol}_rho0.1"] = dict(
                nq=int(len(d)), mean=float(d.mean()), se=se, mde80=float(Z * se),
                ci95=[float(d.mean() - 1.959964 * se), float(d.mean() + 1.959964 * se)])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["cells"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
