"""Round-1 confirmatory analysis (plan: ../preregistration/ROUND1_PLAN.md).
Writes results/tables/round1.json, manuscript/numbers_r1.tex, manuscript/sections/tab_r1.tex, manuscript/figures/fig_r1.pdf."""
import os, json, glob, pickle
import numpy as np
from common import R, D, load_goals
from stats import load
HERE = os.path.dirname(os.path.abspath(__file__)); MS = os.path.join(HERE, "..", "..", "manuscript"); TAB = os.path.join(R, "tables"); os.makedirs(TAB, exist_ok=True)
B = 25; DELTA = 0.02; NBOOT = 10000; RHOS = ["0.1", "1"]; REGS = (("trained", ""), ("zeroshot", "_weak"))
POLS = ["ours", "topm", "thompson", "band", "always"]
num = {}; out = {}
def fmt(x, d=3): return f"{x:.{d}f}"
def sgn(x, d=3): return f"{x:+.{d}f}"
def qid_of(g): return g.rsplit("_h", 1)[0]
def vec(rows, gids, key="gold"): return np.array([rows[g][f"{key}@{B}"] / (2.0 if key == "gold" else 1.0) for g in gids])
def pboot(d, q, seed=0):
    """Paired cluster bootstrap over questions: mean, 95% CI, 90% CI, two-sided p, TOST verdict at DELTA."""
    d = np.asarray(d, float); uq, inv = np.unique(q, return_inverse=True); rng = np.random.default_rng(seed)
    s = np.bincount(inv, d, len(uq)); c = np.bincount(inv, minlength=len(uq)).astype(float)
    idx = rng.integers(0, len(uq), (NBOOT, len(uq))); m = s[idx].sum(1) / c[idx].sum(1)
    p = float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
    lo95, hi95 = np.percentile(m, [2.5, 97.5]); lo90, hi90 = np.percentile(m, [5, 95])
    return dict(mean=float(d.mean()), lo95=float(lo95), hi95=float(hi95), lo90=float(lo90), hi90=float(hi90), p=p,
                equiv=bool(lo90 > -DELTA and hi90 < DELTA), n=int(len(d)), nq=int(len(uq)))
def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj.tolist()
def ci(t): return f"{sgn(t['mean'])} [{sgn(t['lo95'])}, {sgn(t['hi95'])}]"
runs = {}
for d in glob.glob(os.path.join(R, "runs", "test", "*_r1*")):
    n = os.path.basename(d); rows = load("test", n)
    if rows: runs[n] = rows
print(sorted((k, len(v)) for k, v in runs.items()))
for reg, S in REGS:
    base = f"index_r1{S}_rho0.1"
    if base not in runs: continue
    out[reg] = {}
    for rho in RHOS:
        names = {p: f"{p}_r1{S}_rho{rho}" for p in POLS if f"{p}_r1{S}_rho{rho}" in runs}
        if not names: continue
        g = sorted(set(runs[base]).intersection(*[set(runs[n]) for n in names.values()])); q = np.array([qid_of(x) for x in g])
        dep = np.array([int(x[-1]) + 1 for x in g]); v0 = vec(runs[base], g); cell = dict(n=len(g), nq=len(set(q)), scorer=float(v0.mean()))
        for p, n in names.items():
            v = vec(runs[n], g); cell[p] = dict(recall=float(v.mean()), calls=float(vec(runs[n], g, "llm").mean()), vs_scorer=pboot(v - v0, q),
                                                 by_depth={str(k): pboot((v - v0)[dep == k], q[dep == k]) for k in (2, 3, 4)})
            cost = vec(runs[n], g, "fetch") + float(rho) * vec(runs[n], g, "llm"); cell[p]["max_cost"] = float(cost.max())
            if p != "ours" and "ours" in names: cell[p]["ours_minus"] = pboot(vec(runs[names["ours"]], g) - v, q)
        fam = [p for p in ("ours", "topm", "thompson", "band") if p in cell]
        if fam:
            adj = holm([cell[p]["vs_scorer"]["p"] for p in fam])
            for p, a in zip(fam, adj): cell[p]["vs_scorer"]["p_holm"] = a
        fam2 = [p for p in ("topm", "thompson", "band") if p in cell and "ours_minus" in cell[p]]
        if fam2:
            adj = holm([cell[p]["ours_minus"]["p"] for p in fam2])
            for p, a in zip(fam2, adj): cell[p]["ours_minus"]["p_holm"] = a
        out[reg][rho] = cell
# ---------------- macros
RW = {"0.1": "PtOne", "0.5": "PtFive", "1": "One"}; PN = {"ours": "Ours", "topm": "Topm", "thompson": "TS", "band": "Band", "always": "Always"}
for reg, cells in out.items():
    t = "RT" if reg == "trained" else "RZ"
    for rho, cell in cells.items():
        k = RW[rho]; num[f"{t}N{k}"] = str(cell["n"]); num[f"{t}NQ{k}"] = str(cell["nq"]); num[f"{t}Scorer{k}"] = fmt(cell["scorer"])
        for p in POLS:
            if p not in cell: continue
            P = PN[p]; c = cell[p]; num[f"{t}{P}{k}"] = fmt(c["recall"]); num[f"{t}{P}Calls{k}"] = fmt(c["calls"], 1)
            num[f"{t}{P}D{k}"] = ci(c["vs_scorer"]); num[f"{t}{P}DPt{k}"] = sgn(c["vs_scorer"]["mean"])
            num[f"{t}{P}Eq{k}"] = "yes" if c["vs_scorer"]["equiv"] else "no"
            num[f"{t}{P}Ninety{k}"] = f"[{sgn(c['vs_scorer']['lo90'])}, {sgn(c['vs_scorer']['hi90'])}]"
            if "p_holm" in c["vs_scorer"]: num[f"{t}{P}PHolm{k}"] = fmt(c["vs_scorer"]["p_holm"], 3)
            if "ours_minus" in c: num[f"{t}OursMinus{P}{k}"] = ci(c["ours_minus"]); num[f"{t}OursMinus{P}Eq{k}"] = "yes" if c["ours_minus"]["equiv"] else "no"
num["DeltaTOST"] = fmt(DELTA, 2)
# ---------------- table*: both regimes, rho = 0.1 and rho = 1
NAME = {"ours": "\\system", "topm": "Top-$m$ + uncertainty", "thompson": "Thompson over links", "band": "Band cascade"}
def cells_for(reg, p):
    c1 = out.get(reg, {}).get("0.1", {}).get(p); c2 = out.get(reg, {}).get("1", {}).get(p)
    if c1 is None: return ["--"] * 4
    v = c1["vs_scorer"]; eq = "$^{\\equiv}$" if v["equiv"] else ""
    s2 = "--" if c2 is None else f"{sgn(c2['vs_scorer']['mean'])} ({fmt(c2['calls'], 1)})"
    return [fmt(c1["recall"]), f"{sgn(v['mean'])} [{sgn(v['lo95'])}, {sgn(v['hi95'])}]{eq}", fmt(c1["calls"], 1), s2]
if "trained" in out and "0.1" in out["trained"]:
    with open(os.path.join(MS, "sections", "tab_r1.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l@{\\hspace{4pt}}cccc@{\\hspace{8pt}}cccc@{}}\\toprule\n")
        f.write(" & \\multicolumn{4}{c}{trained scorer} & \\multicolumn{4}{c}{zero-shot scorer}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n")
        f.write("Policy & Recall & $\\Delta$ [95\\% CI] & Calls & $\\Delta$ (calls), $\\rho{=}1$ & Recall & $\\Delta$ [95\\% CI] & Calls & $\\Delta$ (calls), $\\rho{=}1$ \\\\\\midrule\n")
        f.write(f"Scorer alone & {fmt(out['trained']['0.1']['scorer'])} & & 0 & & {fmt(out.get('zeroshot', {}).get('0.1', {}).get('scorer', float('nan')))} & & 0 & \\\\\n")
        for p in ("ours", "band", "topm", "thompson"):
            f.write(f"{NAME[p]} & " + " & ".join(cells_for("trained", p) + cells_for("zeroshot", p)) + " \\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
# ---------------- Prediction 2 on the confirmatory crawls (no judge involved): index vs same-head myopic and binary head
conf_q = set(list(dict.fromkeys(g["qid"] for g in load_goals("test")))[80:400])
p2 = {}
idx_old = load("test", "index_rho0.1")
if "index_r1_rho0.1" in runs:     # determinism check: the re-run of the scorer alone matches the round-0 run
    gg = sorted(set(runs["index_r1_rho0.1"]) & set(idx_old)); p2["determinism_identical_frac"] = float(np.mean([runs["index_r1_rho0.1"][x]["gold@25"] == idx_old[x]["gold@25"] for x in gg]))
for my in ("greedy", "greedy_bin"):
    rm = load("test", f"{my}_rho0.1"); g = sorted(x for x in set(idx_old) & set(rm) if qid_of(x) in conf_q); q = np.array([qid_of(x) for x in g]); dep = np.array([int(x[-1]) + 1 for x in g])
    d = vec(idx_old, g) - vec(rm, g); p2[my] = {str(k): pboot(d[dep == k], q[dep == k]) for k in (2, 3, 4)}; p2[my]["all"] = pboot(d, q)
for k, w in ((2, "Two"), (3, "Three"), (4, "Four")):
    num[f"PTwoSameD{w}"] = ci(p2["greedy"][str(k)]); num[f"PTwoSameEq{w}"] = "yes" if p2["greedy"][str(k)]["equiv"] else "no"
    num[f"PTwoBinD{w}"] = ci(p2["greedy_bin"][str(k)])
num["PTwoSameAll"] = ci(p2["greedy"]["all"]); num["PTwoSameAllNinety"] = f"[{sgn(p2['greedy']['all']['lo90'])}, {sgn(p2['greedy']['all']['hi90'])}]"
num["PTwoSameAllEq"] = "yes" if p2["greedy"]["all"]["equiv"] else "no"
out["prediction2"] = p2
# ---------------- redundancy, causal version: gold pages fetched after their link was judged
red = {}
for reg, S in REGS:
    base = f"index_r1{S}_rho0.1"
    if base not in runs: continue
    for p in ("ours", "topm", "thompson", "band"):
        n = f"{p}_r1{S}_rho0.1"
        if n not in runs: continue
        judged_gold = 0; also = 0; gained = 0; lost = 0
        for gid, r in runs[n].items():
            if gid not in runs[base]: continue
            ev = r["events"]; judged = {e[7] for e in ev if e[0] == "I" and e[1] <= B + 1e-9 and len(e) > 7}
            got = {e[6] for e in ev if e[0] == "F" and e[1] <= B + 1e-9 and e[2] == 1 and len(e) > 6}
            got0 = {e[6] for e in runs[base][gid]["events"] if e[0] == "F" and e[1] <= B + 1e-9 and e[2] == 1 and len(e) > 6}
            for pg in got & judged:
                judged_gold += 1; also += int(pg in got0)
            gained += len(got - got0); lost += len(got0 - got)
        red[f"{reg}_{p}"] = dict(judged_gold=judged_gold, also_by_scorer=also, frac_also=also / max(judged_gold, 1), gold_gained=gained, gold_lost=lost, crawls=len(runs[n]))
out["redundancy"] = red
for reg, t in (("trained", "RT"), ("zeroshot", "RZ")):
    for p, P in (("ours", "Ours"), ("topm", "Topm"), ("thompson", "TS"), ("band", "Band")):
        k = f"{reg}_{p}"
        if k in red:
            num[f"{t}{P}JudgedGold"] = str(red[k]["judged_gold"]); num[f"{t}{P}AlsoFrac"] = f"{100 * red[k]['frac_also']:.0f}\\%"
            num[f"{t}{P}Gained"] = str(red[k]["gold_gained"]); num[f"{t}{P}Lost"] = str(red[k]["gold_lost"])
# ---------------- difficulty levels of the splits
lv = {}
for sp in ("train", "cal", "val", "test"):
    gq = {}
    for g in load_goals(sp): gq[g["qid"]] = g["level"]
    lv[sp] = {l: sum(1 for x in gq.values() if x == l) for l in ("easy", "medium", "hard")}
out["levels"] = lv; num["CalHardPct"] = f"{100 * lv['cal']['hard'] / sum(lv['cal'].values()):.0f}\\%"; num["ValHardPct"] = f"{100 * lv['val']['hard'] / sum(lv['val'].values()):.0f}\\%"
num["TestHardPct"] = f"{100 * lv['test']['hard'] / sum(lv['test'].values()):.0f}\\%"
# ---------------- validation-selected configurations
pre = os.path.join(HERE, "..", "preregistration", "round1_baselines.json")
if os.path.exists(pre):
    bj = json.load(open(pre)); out["baselines"] = bj
    for reg, t in (("trained", "RT"), ("zeroshot", "RZ")):
        for rho in RHOS:
            c = bj[reg][rho]; k = RW[rho]
            num[f"{t}TopmCfg{k}"] = f"$({c['topm']['best']['m']},{c['topm']['best']['mu']})$"; num[f"{t}TSCfg{k}"] = f"$\\kappa={c['thompson']['best']['ts_kappa']}$"
            b_ = c["band"]["best"]; num[f"{t}BandCfg{k}"] = f"$[{b_['a']:g}\\lambda, {'\\infty' if b_['b'] > 1e6 else f'{b_['b']:g}\\lambda'}]$"
jl = os.path.join(R, "judge_latency.json")
if os.path.exists(jl):
    J = json.load(open(jl)); num["JudgeSecPerCall"] = fmt(J["seconds_per_call_at_median_throughput"], 2); num["FetchSecPtOne"] = fmt(10 * J["seconds_per_call_at_median_throughput"], 1); num["JudgeMedianMs"] = f"{J['median_call_ms'] / 1000:.1f}"
if "baselines" in out:
    for reg, t in (("trained", "RT"), ("zeroshot", "RZ")):
        for rho in RHOS:
            c = out["baselines"][reg][rho]; k = RW[rho]
            num[f"{t}ValScorer{k}"] = fmt(c["scorer_alone"][0]); num[f"{t}ValOurs{k}"] = fmt(c["twocost"][0]); num[f"{t}ValOursCalls{k}"] = fmt(c["twocost"][1], 1)
json.dump(out, open(os.path.join(TAB, "round1.json"), "w"), indent=1)
with open(os.path.join(MS, "numbers_r1.tex"), "w") as f:
    f.write("% generated by artifact/code/round1_analyze.py\n")
    for k, v in sorted(num.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(num, indent=1))
# ---------------- figure: calibration gain table, exploration test gain, confirmatory gains vs rho
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import crawl as C_
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "legend.fontsize": 5.8, "pdf.fonttype": 42})
COL = {"trained": "#1b4f72", "zeroshot": "#c0392b"}; MOD = {"trained": os.path.join(D, "models"), "zeroshot": os.path.join(D, "models_weak")}
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85)); calinfo = {}
for reg in ("trained", "zeroshot"):
    C_.MODELS = MOD[reg]; os.environ["SF_MODELS"] = MOD[reg]; M = C_.Models(); lam = json.load(open(os.path.join(MOD[reg], "lambda.json")))["25"]["lam"]
    T = M.tables(0.1, lam, conservative="all"); c = M.post["cal"]; gp = C_.chain_index(c["prior"], M.beta)[0]; e = T["edges"]
    b = np.clip(np.searchsorted(e, gp, side="right") - 1, 0, len(e) - 2); keep = T["count"] >= 5
    xs = np.array([gp[b == j].mean() if (b == j).any() else np.nan for j in range(len(e) - 1)]) / lam
    regions = {r: [int(j) for j in np.where(M.tables(float(r), lam, conservative="all")["region"])[0]] for r in RHOS}
    pt_regions = {r: [int(j) for j in np.where(M.tables(float(r), lam)["region"])[0]] for r in RHOS}
    def contiguous(ix): return (not ix) or ix[-1] - ix[0] + 1 == len(ix)
    calinfo[reg] = dict(lam=lam, gain_over_lam=(T["gain"] / lam).tolist(), lcb_over_lam=(T["lcb"] / lam).tolist(), x_over_lam=xs.tolist(), region_final=regions, region_point=pt_regions,
                        final_contiguous={r: contiguous(v) for r, v in regions.items()}, point_contiguous={r: contiguous(v) for r, v in pt_regions.items()})
    ax = axes[0]; ax.plot(xs[keep], (T["gain"] / lam)[keep], marker="o", ms=2.2, color=COL[reg], label=f"{reg.replace('zeroshot', 'zero-shot')}")
    ax.plot(xs[keep], (T["lcb"] / lam)[keep], ls=":", color=COL[reg], lw=0.9, label="lower bound" if reg == "zeroshot" else None)
for r, ls in (("0.1", "--"), ("0.5", "-."), ("1", ":")): axes[0].axhline(float(r), color="#888888", lw=0.6, ls=ls)
axes[0].set_xscale("log"); axes[0].set_yscale("symlog", linthresh=0.5); axes[0].set_yticks([-1, 0, 1, 10]); axes[0].set_yticklabels(["-1", "0", "1", "10"])
axes[0].axvline(1.0, color="#555555", lw=0.6); axes[0].set_xlabel("index / $\\lambda$, calibration"); axes[0].set_ylabel("gain of a call / $\\lambda$"); axes[0].legend(frameon=False, loc="upper left", fontsize=4.6); axes[0].grid(alpha=0.3, lw=0.4)
out["calibration_tables"] = calinfo
for reg, fn in (("trained", "gain_curve.json"), ("zeroshot", "gain_curve_weak.json")):
    p = os.path.join(R, fn)
    if not os.path.exists(p): continue
    G = json.load(open(p)); lam = G["lam"]; xs = np.array([bb["g_mean"] for bb in G["bins"]]) / lam; ys = np.array([bb["gain"] for bb in G["bins"]]) / lam
    lo = np.array([bb["lo"] for bb in G["bins"]]) / lam; hi = np.array([bb["hi"] for bb in G["bins"]]) / lam
    axes[1].errorbar(xs * (1.04 if reg == "zeroshot" else 1.0), ys, yerr=[ys - lo, hi - ys], fmt="o", ms=2.2, color=COL[reg], elinewidth=0.6, capsize=0, label=f"{reg.replace('zeroshot', 'zero-shot')}")
axes[1].set_xscale("log"); axes[1].set_yscale("symlog", linthresh=0.5); axes[1].set_yticks([-10, -1, 0, 1, 10]); axes[1].set_yticklabels(["-10", "-1", "0", "1", "10"]); axes[1].axvline(1.0, color="#555555", lw=0.6)
axes[1].set_xlabel("index / $\\lambda$, held-out links"); axes[1].legend(frameon=False, loc="upper left", fontsize=4.6); axes[1].grid(alpha=0.3, lw=0.4)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(MS, "figures", "fig_r1.pdf")); plt.close(fig)
for reg, t in (("trained", "RT"), ("zeroshot", "RZ")):
    ci_ = calinfo[reg]
    for r in RHOS: k = RW[r]; num[f"{t}FinalBins{k}"] = str(len(ci_["region_final"][r])); num[f"{t}PointBins{k}"] = str(len(ci_["region_point"][r]))
    num[f"{t}FinalContig"] = "yes" if all(ci_["final_contiguous"].values()) else "no"; num[f"{t}PointContig"] = "yes" if all(ci_["point_contiguous"].values()) else "no"
json.dump(out, open(os.path.join(TAB, "round1.json"), "w"), indent=1)
with open(os.path.join(MS, "numbers_r1.tex"), "w") as f:
    f.write("% generated by artifact/code/round1_analyze.py\n")
    for k, v in sorted(num.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
