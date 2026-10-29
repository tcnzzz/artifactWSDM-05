"""Round 1: choose top-m, Thompson-over-links and band configurations per regime and rho on validation questions 20..69
(highest mean recall at cost 25, ties within 0.001 to fewer calls). Writes preregistration/round1_baselines.json and appends
its sha256 and a timestamp to preregistration/round1_rule_hashes.txt. Must run before round1_test.sh."""
import os, json, glob, hashlib, time
import numpy as np
from common import R
from stats import load
HERE = os.path.dirname(os.path.abspath(__file__)); PRE = os.path.join(HERE, "..", "preregistration"); B = 25
def score(name):
    rows = load("val", name)
    if len(rows) < 150: return None
    return float(np.mean([r[f"gold@{B}"] / 2 for r in rows.values()])), float(np.mean([r[f"llm@{B}"] for r in rows.values()])), len(rows)
out = {}
for reg, S in (("trained", ""), ("zeroshot", "_weak")):
    out[reg] = {}
    idx = score(f"index_r1{S}_rho0.1")
    for rho in ("0.1", "1"):
        cell = {"scorer_alone": idx, "twocost": score(f"ours_r1{S}_rho{rho}")}
        grids = {"topm": [(f"_m{m}u{u}", {"m": m, "mu": u}) for m in (0, 1, 3, 5) for u in (0, 1) if (m, u) != (0, 0)],
                 "thompson": [(f"_k{k}", {"ts_kappa": k}) for k in (3, 10, 30)],
                 "band": [(f"_a{a}b{b}", {"a": a, "b": b}) for a in (0.25, 0.5, 1) for b in (2, 1000000000) if a < b]}
        for pol, grid in grids.items():
            res = []
            for tag, params in grid:
                s = score(f"{pol}_r1{S}{tag}_rho{rho}")
                if s is None: raise SystemExit(f"missing or incomplete validation run {pol}_r1{S}{tag}_rho{rho}")
                res.append((s[0], -s[1], tag, params, s))
            if pol == "topm": res.append((idx[0], 0.0, "_m0u0", {"m": 0, "mu": 0}, idx))
            best = max(res, key=lambda t: t[0]); tied = [t for t in res if t[0] >= best[0] - 0.001]; best = max(tied, key=lambda t: t[1])
            cell[pol] = dict(best=best[3], best_tag=best[2], val_recall=best[4][0], val_calls=best[4][1], grid={t[2]: dict(recall=t[4][0], calls=t[4][1]) for t in res})
        out[reg][rho] = cell
p = os.path.join(PRE, "round1_baselines.json"); json.dump(out, open(p, "w"), indent=1)
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
with open(os.path.join(PRE, "round1_rule_hashes.txt"), "a") as f: f.write(f"{h}  preregistration/round1_baselines.json  (written {time.strftime('%Y-%m-%d %H:%M:%S %z')})\n")
for reg in out:
    for rho, c in out[reg].items(): print(reg, rho, "scorer", round(c["scorer_alone"][0], 3), "twocost", c["twocost"] and round(c["twocost"][0], 3), {p_: (c[p_]["best_tag"], round(c[p_]["val_recall"], 3), round(c[p_]["val_calls"], 1)) for p_ in ("topm", "thompson", "band")})
