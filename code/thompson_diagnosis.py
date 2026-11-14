"""Where do judge calls go? A diagnostic on the completed confirmatory blocks (80--199).

Reads only saved run files under results/runs/test.  For every judge call we ask whether the
inspected link was later fetched in the same crawl, whether it was a gold page, and whether it
was within the horizon (true layer <= 2).  This separates a call that contributes to a skip from
a call that pays for a link the crawler then follows anyway, and shows why the untuned policies
that keep calling at rho=1 lose recall: they spend the budget on links they then reject.

No new judge call; no run is modified.  Writes results/tables/thompson_diagnosis.json.
"""
import glob
import json
import os

import numpy as np

from common import R, load_goals
from stats import load

OUT = os.path.join(R, "tables", "thompson_diagnosis.json")
B = 25
CONF_LO, CONF_HI = 80, 200
POLS = ("ours", "topm", "thompson", "band", "always")


def qid_of(gid):
    return gid.rsplit("_h", 1)[0]


def cell(run, gids):
    insp = fetched = insp_fetched = insp_skip = gold = lead = 0
    calls = []
    for g in gids:
        r = run[g]
        ev = r["events"]
        fs = {int(e[6]) for e in ev if e[0] == "F"}
        ii = [e for e in ev if e[0] == "I"]
        calls.append(len(ii))
        for e in ii:
            insp += 1
            tgt = int(e[7])
            if tgt in fs:
                insp_fetched += 1
            else:
                insp_skip += 1
            gold += int(e[2])
            lead += int(e[3] <= 2)
        fetched += len(fs)
    return dict(n_crawls=len(gids), calls=float(np.mean(calls)), max_calls=int(np.max(calls)),
                insp=int(insp), insp_fetched=int(insp_fetched), insp_skip=int(insp_skip),
                frac_fetched=float(insp_fetched / max(insp, 1)),
                frac_saved=float(insp_skip / max(insp, 1)),
                frac_gold=float(gold / max(insp, 1)), frac_on_path=float(lead / max(insp, 1)),
                fetch_per_crawl=float(fetched / max(len(gids), 1)))


def main():
    order, seen = [], set()
    for g in load_goals("test"):
        if g["qid"] not in seen:
            seen.add(g["qid"])
            order.append(g["qid"])
    pos = {q: i for i, q in enumerate(order)}
    conf_q = {q for q, i in pos.items() if CONF_LO <= i < CONF_HI}

    out = {"config": {"blocks": [80, 120, 160], "questions": len(conf_q), "budget": B,
                      "note": "diagnostic on completed confirmatory blocks; no new judge call"},
           "cells": {}}
    for reg, S in (("trained", ""), ("zeroshot", "_weak")):
        base = f"index_r1{S}_rho0.1"
        if not os.path.isdir(os.path.join(R, "runs", "test", base)):
            continue
        out["cells"][reg] = {}
        for rho in ("0.1", "1"):
            names = {}
            for p in POLS:
                # 'always' runs are not r1-tagged
                cand = [f"{p}_r1{S}_rho{rho}", f"{p}{S}_rho{rho}"]
                for n in cand:
                    if os.path.isdir(os.path.join(R, "runs", "test", n)):
                        names[p] = n
                        break
            if not names:
                continue
            loaded = {p: load("test", n) for p, n in names.items()}
            base_rows = load("test", base)
            bq = [r for g, r in base_rows.items() if qid_of(g) in conf_q and 2 <= r["depth"] <= 4]
            c = {"rho": float(rho),
                 "scorer": {"recall": float(np.mean([r["gold@25"] / 2 for r in bq]))}}
            # the four judge policies share a paired question set; 'always' was run on an earlier
            # 80-question set and is reported on its own intersection.
            main_pols = [p for p in ("ours", "topm", "thompson", "band") if p in loaded]
            common = set(base_rows)
            for p in main_pols:
                common &= set(loaded[p])
            byq = {}
            for g in common:
                byq.setdefault(qid_of(g), set()).add(base_rows[g]["depth"])
            keep_q = {q for q, ds in byq.items() if ds == {2, 3, 4}}
            gids_main = sorted(g for g in common if qid_of(g) in keep_q)
            for p, run in loaded.items():
                if p in main_pols:
                    gids = gids_main
                else:
                    common_p = set(base_rows) & set(run)
                    byq_p = {}
                    for g in common_p:
                        byq_p.setdefault(qid_of(g), set()).add(base_rows[g]["depth"])
                    keep_p = {q for q, ds in byq_p.items() if ds == {2, 3, 4}}
                    gids = sorted(g for g in common_p if qid_of(g) in keep_p)
                if not gids:
                    continue
                rec = cell(run, gids)
                rec["recall"] = float(np.mean([run[g]["gold@25"] / 2 for g in gids]))
                rec["n"] = len(gids)
                rec["nq"] = len({qid_of(g) for g in gids})
                c[p] = rec
            out["cells"][reg][rho] = c
            print(reg, "rho", rho)
            for p in POLS:
                if p not in c:
                    continue
                r = c[p]
                print(f"  {p:9s} n={r['n']:3d} q={r['nq']:3d} recall {r['recall']:.3f} calls {r['calls']:5.2f} "
                      f"insp-fetched {r['frac_fetched']:.2f} skipped {r['frac_saved']:.2f} "
                      f"gold {r['frac_gold']:.3f} on-path {r['frac_on_path']:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
