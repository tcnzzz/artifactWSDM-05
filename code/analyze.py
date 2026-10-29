"""Produce every number, table and figure of the paper from the raw run files in results/runs/ and the fitted models.
Writes results/tables/tables.json, manuscript/numbers.tex, manuscript/sections/tab_main.tex, manuscript/figures/*.pdf."""
import os, json, glob, pickle
import numpy as np
from common import R, D, load_goals
from crawl import chain_index, K
from stats import load, boot
HERE = os.path.dirname(os.path.abspath(__file__)); MS = os.path.join(HERE, "..", "..", "manuscript"); TAB = os.path.join(R, "tables"); os.makedirs(TAB, exist_ok=True)
FIG = os.path.join(MS, "figures"); os.makedirs(FIG, exist_ok=True)
MOD = {"strong": os.path.join(D, "models"), "weak": os.path.join(D, "models_weak")}
SUF = {"strong": "", "weak": "_weak"}
RHO = 0.1; BH = 25; BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
num = {"HeadlineBudget": str(25)}; tables = {}
def fmt(x, d=3): return f"{x:.{d}f}"
def ci(t, d=3): return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
def sci(t, d=3): return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"
def RW(rho): return {0.05: "PtZeroFive", 0.1: "PtOne", 0.2: "PtTwo", 0.5: "PtFive", 1.0: "One"}[round(float(rho), 2)]
def qid_of(g): return g.rsplit("_h", 1)[0]
def dep_of(g): return int(g[-1]) + 1
runs = {}
for d in sorted(glob.glob(os.path.join(R, "runs", "test", "*"))):
    rows = load("test", os.path.basename(d))
    if rows: runs[os.path.basename(d)] = rows
def common(names):
    s = None
    for n in names: s = set(runs[n]) if s is None else s & set(runs[n])
    return sorted(s)
def vec(name, gids, B, key="gold"):
    return np.array([runs[name][g][f"{key}@{B}"] / (2.0 if key == "gold" else 1.0) for g in gids])
def L(x): return float(json.load(open(os.path.join(MOD[x], "lambda.json")))["25"]["lam"])
# ---------------------------------------------------------------- graph and data facts
st = json.load(open(os.path.join(D, "graph", "stats.json"))); num["NPages"] = fmt(st["pages"] / 1e6, 2); num["NEdges"] = fmt(st["edges_dedup"] / 1e6, 2)
num["OutDegMean"] = fmt(st["out_degree_mean"], 2); gs = json.load(open(os.path.join(D, "goals_stats.json")))
for k in ("test", "train", "cal", "val"): num["NQ" + k.capitalize()] = str(gs[k])
cal = pickle.load(open(os.path.join(MOD["strong"], "calib.pkl"), "rb")); num["BetaMain"] = fmt(cal["beta"], 2); num["LambdaMain"] = fmt(L("strong"), 4)
if os.path.exists(os.path.join(MOD["weak"], "lambda.json")):
    num["LambdaWeak"] = fmt(L("weak"), 4); num["BetaWeak"] = fmt(pickle.load(open(os.path.join(MOD["weak"], "calib.pkl"), "rb"))["beta"], 2)
pi = json.load(open(os.path.join(MOD["strong"], "posterior_info.json")))
num["AUCJudge"] = fmt(pi["R1"]["auc_llm"]); num["AUCEncoder"] = fmt(pi["R1"]["auc_prior"]); num["AUCPost"] = fmt(pi["R1"]["auc_post_oof"]); num["NCalLinks"] = f"{pi['n']:,}".replace(",", "{,}")
if os.path.exists(os.path.join(MOD["weak"], "posterior_info.json")):
    pw = json.load(open(os.path.join(MOD["weak"], "posterior_info.json"))); num["AUCWeakPrior"] = fmt(pw["R1"]["auc_prior"]); num["AUCWeakPost"] = fmt(pw["R1"]["auc_post_oof"]); num["AUCWeakJudge"] = fmt(pw["R1"]["auc_llm"])
    num["NCalLinksWeak"] = f"{pw['n']:,}".replace(",", "{,}"); tables["posterior_info_weak"] = pw
v0 = json.load(open(os.path.join(MOD["strong"], "posterior_info_v0.json"))); num["VZeroMaxGain"] = fmt(max(v0["realised_gain_over_lam_by_bin"]), 3)
tables["posterior_info"] = pi; tables["posterior_info_v0"] = v0
# ---------------------------------------------------------------- non-judge policies on all test crawls
NONLLM = ["bfs", "pagerank", "bm25", "dense", "coverage", "greedy", "greedy_bin", "reach", "index"]
full = [p for p in NONLLM if f"{p}_rho0.1" in runs]; gall = common([f"{p}_rho0.1" for p in full]); qall = np.array([qid_of(g) for g in gall]); dall = np.array([dep_of(g) for g in gall])
num["NTestGoals"] = str(len(gall)); tables["full"] = {}
for p in full:
    tables["full"][p] = {f"r{B}": boot(vec(f"{p}_rho0.1", gall, B), qall) for B in (10, 25, 50)}
    tables["full"][p].update({f"r{BH}_d{d}": boot(vec(f"{p}_rho0.1", gall, BH)[dall == d], qall[dall == d]) for d in (2, 3, 4)})
def full_macro(name, p, B=BH): num[name] = fmt(tables["full"][p][f"r{B}"][0])
for name, p in [("FullBFS", "bfs"), ("FullPR", "pagerank"), ("FullBM", "bm25"), ("FullDense", "dense"), ("FullCov", "coverage"), ("FullGreedy", "greedy"), ("FullIndex", "index"), ("FullReach", "reach")]:
    if p in tables["full"]: full_macro(name, p)
# index vs myopic by depth (Prediction 2), paired, all crawls
gd = {}
for my in ("greedy", "greedy_bin"):
    if my not in full: continue
    for B in (10, BH, 50):
        diff = vec("index_rho0.1", gall, B) - vec(f"{my}_rho0.1", gall, B)
        for d in (2, 3, 4): gd[f"{my}_B{B}_d{d}"] = boot(diff[dall == d], qall[dall == d])
tables["index_minus_myopic"] = gd
if gd:
    for d, w in ((2, "Two"), (3, "Three"), (4, "Four")): num[f"IdxMyopicD{w}"] = sci(gd[f"greedy_bin_B{BH}_d{d}"])
diag = os.path.join(TAB, "diag_index.json")
if os.path.exists(diag):
    dj = json.load(open(diag)); num["SameTopFrac"] = pct = f"{100 * dj['frac_same_top_link']:.1f}\\%"
    num["GateRankIndex"] = fmt(dj["gateway_links"]["mean_rank_index"], 1); num["GateRankMyopic"] = fmt(dj["gateway_links"]["mean_rank_pi1"], 1)
    num["GateIdxBetter"] = f"{100 * dj['gateway_links']['frac_index_rank_better']:.1f}\\%"
# ---------------------------------------------------------------- judge policies, both regimes, common crawls
# "ours_cons" is TwoCost (conservative region); "ours" is the point-estimate region, kept as an ablation.
JP = ["index", "always", "topm", "thompson", "ours", "ours_cons"]
PN = {"index": "Index", "always": "Always", "topm": "Topm", "thompson": "TS", "ours": "Point", "ours_cons": "Ours", "llm_only": "LLMonly"}
reg_rows = {}
for reg in ("strong", "weak"):
    s = SUF[reg]; names = [n for n in [f"{p}{s}_rho0.1" for p in JP] if n in runs]
    if f"ours_cons{s}_rho0.1" not in runs: continue
    g = common(names); q = np.array([qid_of(x) for x in g]); dd = np.array([dep_of(x) for x in g]); out = {"n": len(g), "nq": len(set(q))}
    for n in names:
        p = n[: -len(s + "_rho0.1")] if s else n[: -len("_rho0.1")]
        out[p] = dict(r10=boot(vec(n, g, 10), q), r25=boot(vec(n, g, BH), q), r25_d4=boot(vec(n, g, BH)[dd == 4], q[dd == 4]),
                      calls=float(vec(n, g, BH, "llm").mean()), fetch=float(vec(n, g, BH, "fetch").mean()),
                      minus_index=boot(vec(n, g, BH) - vec(f"index{s}_rho0.1", g, BH), q))
        if p != "ours_cons": out[p]["cons_minus"] = boot(vec(f"ours_cons{s}_rho0.1", g, BH) - vec(n, g, BH), q)
    reg_rows[reg] = out
if "llm_only_rho0.1" in runs and "ours_cons_rho0.1" in runs:     # LLM-only frontier: its own (smaller) subset, paired
    g = common(["llm_only_rho0.1", "ours_cons_rho0.1", "index_rho0.1"]); q = np.array([qid_of(x) for x in g])
    reg_rows["llm_only"] = dict(n=len(g), nq=len(set(q)), r25=boot(vec("llm_only_rho0.1", g, BH), q), calls=float(vec("llm_only_rho0.1", g, BH, "llm").mean()),
                                minus_index=boot(vec("llm_only_rho0.1", g, BH) - vec("index_rho0.1", g, BH), q))
    R_ = reg_rows["llm_only"]; num["LLMonlyN"] = str(R_["n"]); num["SLLMonly"] = fmt(R_["r25"][0]); num["SLLMonlyCalls"] = fmt(R_["calls"], 1); num["SLLMonlyMinusIdx"] = sci(R_["minus_index"])
tables["judge"] = reg_rows
for reg in ("strong", "weak"):
    if reg not in reg_rows: continue
    R_ = reg_rows[reg]; tag = "S" if reg == "strong" else "W"; num[f"N{tag}Goals"] = str(R_["n"]); num[f"N{tag}Q"] = str(R_["nq"])
    for p in JP:
        if p not in R_: continue
        P = PN[p]; num[f"{tag}{P}"] = fmt(R_[p]["r25"][0]); num[f"{tag}{P}Calls"] = fmt(R_[p]["calls"], 1)
        if p != "index": num[f"{tag}{P}MinusIdx"] = sci(R_[p]["minus_index"]); num[f"{tag}{P}MinusIdxPoint"] = f"{R_[p]['minus_index'][0]:+.3f}"
        if "cons_minus" in R_[p]: num[f"{tag}OursMinus{P}"] = sci(R_[p]["cons_minus"])
# ---------------------------------------------------------------- main table
NAME = dict(bfs="BFS", pagerank="PageRank (full graph)", bm25="BM25 anchor", dense="Dense similarity", coverage="Greedy coverage",
            greedy="Scorer, myopic", greedy_bin="Scorer, myopic (binary head)", reach="Scorer, reach $R_K$", index="Scorer alone (index)",
            llm_only="LLM-only frontier", always="Judge before every fetch", topm="Top-$m$ + uncertainty", thompson="Thompson sampling",
            ours="\\system, point-estimate region", ours_cons="\\system")
if "strong" in reg_rows:
    S_ = reg_rows["strong"]; W_ = reg_rows.get("weak", {}); gS = common([f"{p}_rho0.1" for p in JP if f"{p}_rho0.1" in runs])
    qS = np.array([qid_of(x) for x in gS])
    def cell(R_, p):
        if p not in R_: return "-- & --"
        return f"{fmt(R_[p]['r25'][0])} & {fmt(R_[p]['calls'], 1)}"
    with open(os.path.join(MS, "sections", "tab_main.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lcccc@{}}\\toprule\n & \\multicolumn{2}{c}{trained scorer} & \\multicolumn{2}{c}{zero-shot scorer}\\\\\n")
        f.write("\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\nPolicy & R@%d & calls & R@%d & calls \\\\\\midrule\n" % (BH, BH))
        for p in ["bfs", "bm25", "coverage", "greedy"]:
            if f"{p}_rho0.1" in runs: f.write(f"{NAME[p]} & {fmt(boot(vec(f'{p}_rho0.1', gS, BH), qS)[0])} & 0 & -- & -- \\\\\n")
        f.write(f"{NAME['index']} & {cell(S_, 'index')} & {cell(W_, 'index')} \\\\\n\\midrule\n")
        for p in ["always", "thompson", "topm", "ours", "ours_cons"]:
            nm = NAME[p] if p != "ours_cons" else "\\textbf{" + NAME[p] + "}"
            f.write(f"{nm} & {cell(S_, p)} & {cell(W_, p)} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
# ---------------------------------------------------------------- rho sweep, measured and predicted judge calls
def predicted_calls(reg, rho, lam, B, conservative=False):
    """Calls per crawl predicted from calibration crawls of the scorer-only index policy (no test data, no judge on test):
    links seen within the first B fetches whose prior bin is in the escalation region and whose ranking key exceeds lam."""
    import crawl as C_; os.environ["SF_MODELS"] = MOD[reg]; C_.MODELS = MOD[reg]
    M = C_.Models(); T = M.tables(rho, lam, conservative=conservative); goals = {g["gid"]: g for g in load_goals("cal")}; kind = "index_C" if reg == "strong" else "index_weakC"
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "rows", f"cal_{kind}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
    calls = []
    for r in rows:
        m = r["in_crawl"]; pr = r["prior"][m]; par = r["par"][m]; g = chain_index(pr, M.beta)[0]
        allowed = set([goals[r["gid"]]["seed"]] + [int(x) for x in r["fetched"][:B]])
        b = np.clip(np.searchsorted(T["edges"], g, side="right") - 1, 0, len(T["edges"]) - 2); key = np.maximum(g, T["insp_index"][b])
        sel = np.array([p_ in allowed for p_ in par]) & T["region"][b] & (key > lam); calls.append(int(sel.sum()))
    return float(np.mean(calls)), T
sweep = {}
for reg in ("strong", "weak"):
    s = SUF[reg]
    if not os.path.exists(os.path.join(MOD[reg], "lambda.json")): continue
    lam = L(reg); idx = f"index{s}_rho0.1"
    for var in ("ours", "ours_cons"):
        rs = sorted(set(float(n.split("_rho")[1]) for n in runs if n.startswith(f"{var}{s}_rho")))
        out = []
        for rho in rs:
            n = f"{var}{s}_rho{rho:g}"; g = sorted(runs[n]); q = np.array([qid_of(x) for x in g])
            row = dict(rho=rho, n=len(g), r=boot(vec(n, g, BH), q), calls=float(vec(n, g, BH, "llm").mean()))
            pc, T = predicted_calls(reg, rho, lam, BH, conservative=(var == "ours_cons")); row["pred_calls"] = pc; row["region_bins"] = int(T["region"].sum())
            row["index"] = boot(vec(idx, g, BH), q); row["minus_index"] = boot(vec(n, g, BH) - vec(idx, g, BH), q)
            if var == "ours_cons":
                for hp in ("always", "topm"):
                    al = f"{hp}{s}_rho{rho:g}"
                    if al in runs:
                        g2 = common([n, al, idx]); q2 = np.array([qid_of(x) for x in g2]); row[f"{hp}_calls"] = float(vec(al, g2, BH, "llm").mean())
                        row[f"{hp}_minus_index"] = boot(vec(al, g2, BH) - vec(idx, g2, BH), q2)
            out.append(row)
        sweep[f"{reg}_{var}"] = out
tables["sweep"] = sweep
for key, out in sweep.items():
    reg, var = key.split("_", 1); tag = ("S" if reg == "strong" else "W") + ("" if var == "ours_cons" else "Pt")
    for r in out:
        k = RW(r["rho"])
        num[f"{tag}Calls{k}"] = fmt(r["calls"], 1); num[f"{tag}PredCalls{k}"] = fmt(r["pred_calls"], 1); num[f"{tag}Region{k}"] = str(r["region_bins"])
        num[f"{tag}MinusIdx{k}"] = sci(r["minus_index"])
        for hp in ("always", "topm"):
            if f"{hp}_minus_index" in r: num[f"{tag}{hp.capitalize()}MinusIdx{k}"] = sci(r[f"{hp}_minus_index"]); num[f"{tag}{hp.capitalize()}Calls{k}"] = fmt(r[f"{hp}_calls"], 1)
# ---------------------------------------------------------------- held-out gain curves
for reg, fn in (("strong", "gain_curve.json"), ("weak", "gain_curve_weak.json")):
    p = os.path.join(R, fn)
    if not os.path.exists(p): continue
    G = json.load(open(p)); tag = "S" if reg == "strong" else "W"; tables[f"gain_{reg}"] = G
    num[f"{tag}GainN"] = f"{G['n']:,}".replace(",", "{,}"); num[f"{tag}GainPeak"] = fmt(G["peak_g_mean"], 4)
    num[f"{tag}GainInterval"] = "all" if all(G["realised_region_is_interval"].values()) else ", ".join(r for r, v in G["realised_region_is_interval"].items() if not v)
    num[f"{tag}GainPeakOverLam"] = fmt(max(b["gain"] for b in G["bins"]) / G["lam"], 2)
    for r, v in G["region_width_bins"].items(): num[f"{tag}Width" + RW(float(r))] = f"{v['realised']}"
for reg, fn in (("strong", "gain_curve.json"), ("weak", "gain_curve_weak.json")):
    p = os.path.join(R, fn)
    if not os.path.exists(p): continue
    G = json.load(open(p)); tag = "S" if reg == "strong" else "W"; lam_ = G["lam"]
    reg_bins = [b for b in G["bins"] if b["realised_region"]["0.1"]]
    if reg_bins: num[f"{tag}GainLo"] = fmt(min(b["g_lo"] for b in reg_bins) / lam_, 2); num[f"{tag}GainHi"] = fmt(max(b["g_hi"] for b in reg_bins) / lam_, 2)
    num[f"{tag}GainPeakAt"] = fmt(G["peak_g_mean"] / lam_, 2)
if "strong_ours_cons" in sweep:
    r1 = [r for r in sweep["strong_ours_cons"] if r["rho"] == 1.0]
    if r1 and "always_minus_index" in r1[0]: num["SAlwaysMinusIdxOneAbs"] = fmt(-r1[0]["always_minus_index"][0])
if "weak_ours_cons" in sweep:
    act = [r for r in sweep["weak_ours_cons"] if r["calls"] > 0]
    if act: num["WConsMinMinus"] = f"{min(r['minus_index'][0] for r in act):+.3f}"; num["WConsMaxMinus"] = f"{max(r['minus_index'][0] for r in act):+.3f}"
if "strong_ours" in sweep:
    act = [r for r in sweep["strong_ours"] if r["calls"] > 0]
    if act: num["SPtPredRatioMax"] = fmt(max(r["pred_calls"] / r["calls"] for r in act), 1); num["SPtPredRatioMin"] = fmt(min(r["pred_calls"] / r["calls"] for r in act), 1)
_loss = [-r[f"{hp}_minus_index"][0] for k_ in ("strong_ours_cons", "weak_ours_cons") for r in sweep.get(k_, []) for hp in ("always", "topm") if f"{hp}_minus_index" in r]
if _loss: num["HeurMaxLoss"] = fmt(max(_loss))
if "weak" in reg_rows and "ours_cons" in reg_rows["weak"]: num["WOursMinusIdxPoint"] = f"{reg_rows['weak']['ours_cons']['minus_index'][0]:+.3f}"
import sqlite3
_db = sqlite3.connect(os.path.join(D, "cache", "llm.sqlite")); _n, _t = _db.execute("SELECT count(*), sum(ptok) FROM insp").fetchone(); _db.close()
json.dump(dict(distinct_judge_calls=int(_n), prompt_tokens=int(_t or 0), model="qwen2.5:7b", hosted_api_usd=0.0), open(os.path.join(R, "llm_usage.json"), "w"), indent=1)
num["NJudgeCalls"] = f"{_n:,}".replace(",", "{,}"); num["NJudgePromptTok"] = fmt((_t or 0) / 1e6, 1)
rd = os.path.join(TAB, "diag_redundancy.json")
if os.path.exists(rd):
    rj = json.load(open(rd)); num["RedundLOne"] = fmt(rj["index_T"]["mean_by_layer"]["1"], 1); num["RedundLTwo"] = fmt(rj["index_T"]["mean_by_layer"]["2"], 1)
    num["RedundFrac"] = f"{100 * rj['index_T']['frac_crawls_ge2_layer1']:.0f}\\%"; num["RedundLOneW"] = fmt(rj["index_weakT"]["mean_by_layer"]["1"], 1)
# ---------------------------------------------------------------- ablations (strong regime, rho = 0.1)
abl = {}
for n in ("ours_lamx0.5_rho0.1", "ours_lamx2_rho0.1", "ours_online_rho0.1"):
    if n in runs and "ours_rho0.1" in runs:
        g = common([n, "ours_rho0.1"]); q = np.array([qid_of(x) for x in g])
        abl[n] = dict(n=len(g), r=boot(vec(n, g, BH), q), diff=boot(vec(n, g, BH) - vec("ours_rho0.1", g, BH), q), calls=float(vec(n, g, BH, "llm").mean()))
tables["ablations"] = abl
for n, k in (("ours_lamx0.5_rho0.1", "LamHalf"), ("ours_lamx2_rho0.1", "LamTwice"), ("ours_online_rho0.1", "Online")):
    if n in abl: num[k + "Diff"] = sci(abl[n]["diff"])
for n in runs:                      # cross-model replication cells, if present
    if n.endswith("_llama_rho0.1"): pass
rep = {}
for p in ("ours", "topm", "always"):
    n = f"{p}_llama_rho0.1"
    if n in runs: rep[p] = n
if len(rep) >= 2:
    g = common(list(rep.values()) + ["index_rho0.1"]); q = np.array([qid_of(x) for x in g]); tables["replication"] = {p: dict(r=boot(vec(n, g, BH), q), calls=float(vec(n, g, BH, "llm").mean())) for p, n in rep.items()}
    tables["replication"]["index"] = dict(r=boot(vec("index_rho0.1", g, BH), q)); num["ReplN"] = str(len(g))
    for p in rep: num["Repl" + {"ours": "Ours", "topm": "Topm", "always": "Always"}[p]] = fmt(tables["replication"][p]["r"][0])
    num["ReplIndex"] = fmt(tables["replication"]["index"]["r"][0])
spend = os.path.join(R, "llm_usage.json")
# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "legend.fontsize": 6, "pdf.fonttype": 42, "axes.titlesize": 7.5})
COL = dict(index="#1b4f72", greedy_bin="#f39c12", bm25="#7d3c98", bfs="#7f8c8d", coverage="#148f77", reach="#2e86c1", ours="#c0392b", always="#7f8c8d", strong="#1b4f72", weak="#c0392b")
LAB = dict(index="scorer, index", greedy_bin="scorer, myopic", bm25="BM25 anchor", bfs="BFS", coverage="greedy coverage", reach="scorer, reach")
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75), sharey=True)
for ax, d in zip(axes, (2, 3, 4)):
    sel = dall == d
    for p in ("bfs", "bm25", "coverage", "greedy_bin", "index"):
        if p not in full: continue
        ys = []; lo = []; hi = []
        for B in BUDGETS:
            m_, l_, h_ = boot(vec(f"{p}_rho0.1", gall, B)[sel], qall[sel], n=500); ys.append(m_); lo.append(l_); hi.append(h_)
        ax.plot(BUDGETS, ys, color=COL[p], label=LAB[p], ls="--" if p == "greedy_bin" else "-"); ax.fill_between(BUDGETS, lo, hi, color=COL[p], alpha=0.13, lw=0)
    ax.set_title(f"target depth {d}"); ax.set_xlabel("budget (fetches)"); ax.grid(alpha=0.3, lw=0.4)
axes[0].set_ylabel("recall of gold pages"); axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "fig_depth.pdf")); plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85))
ax = axes[0]
for reg, fn in (("strong", "gain_curve.json"), ("weak", "gain_curve_weak.json")):
    p = os.path.join(R, fn)
    if not os.path.exists(p): continue
    G = json.load(open(p)); xs = np.array([b["g_mean"] for b in G["bins"]]) / G["lam"]; ys = np.array([b["gain"] for b in G["bins"]]) / G["lam"]
    ax.plot(xs, ys, marker="o", ms=2.2, color=COL[reg], label=f"{'trained' if reg == 'strong' else 'zero-shot'} scorer, test")
    ax.fill_between(xs, np.array([b["lo"] for b in G["bins"]]) / G["lam"], np.array([b["hi"] for b in G["bins"]]) / G["lam"], color=COL[reg], alpha=0.15, lw=0)
    ax.plot(xs, np.array([b["predicted_gain"] for b in G["bins"]]) / G["lam"], ls=":", color=COL[reg], lw=0.9)
ax.axvline(1.0, color="#555555", lw=0.6); ax.axhline(0.1, color="#999999", lw=0.6, ls="--")
ax.set_yscale("symlog", linthresh=0.5); ax.set_xlim(left=0.05); ax.set_ylim(-40, 120); ax.set_yticks([-10, -1, 0, 1, 10, 100]); ax.set_yticklabels(["-10", "-1", "0", "1", "10", "100"])
ax.set_xscale("log"); ax.set_xlabel(r"scorer index / fetch price $\lambda$"); ax.set_ylabel(r"gain of a judge call / $\lambda$"); ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, loc="upper left")
ax = axes[1]
for reg in ("strong", "weak"):
    out = sweep.get(f"{reg}_ours", [])
    if not out: continue
    xs = [r["rho"] for r in out]; ax.plot(xs, [r["calls"] for r in out], marker="o", ms=3, color=COL[reg], label=f"{'trained' if reg == 'strong' else 'zero-shot'}, measured")
    ax.plot(xs, [r["pred_calls"] for r in out], marker="s", ms=2.5, ls=":", color=COL[reg], label="predicted")
ax.set_xscale("log"); ax.set_xlabel(r"cost ratio $\rho = c_l/c_f$"); ax.set_ylabel(f"judge calls per crawl (B={BH})"); ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False)
ax = axes[2]
for reg in ("strong", "weak"):
    out = sweep.get(f"{reg}_ours_cons", [])
    if not out: continue
    xs = [r["rho"] for r in out]; lab = "trained" if reg == "strong" else "zero-shot"
    ax.plot(xs, [r["minus_index"][0] for r in out], marker="o", ms=3, color=COL[reg], label=f"TwoCost ({lab})")
    ax.fill_between(xs, [r["minus_index"][1] for r in out], [r["minus_index"][2] for r in out], color=COL[reg], alpha=0.12, lw=0)
    for hp, mk, ls_ in (("always", "x", "--"), ("topm", "^", ":")):
        xa = [r["rho"] for r in out if f"{hp}_minus_index" in r]
        if xa: ax.plot(xa, [r[f"{hp}_minus_index"][0] for r in out if f"{hp}_minus_index" in r], marker=mk, ms=3, ls=ls_, color=COL[reg], lw=0.8,
                       label={"always": "judge every fetch", "topm": "top-$m$"}[hp] + f" ({lab})")
ax.axhline(0, color="#555555", lw=0.6); ax.set_xscale("log"); ax.set_xlabel(r"cost ratio $\rho$"); ax.set_ylabel(f"recall gain over scorer (B={BH})"); ax.grid(alpha=0.3, lw=0.4); ax.legend(frameon=False, fontsize=5.2, ncol=1)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "fig_rho.pdf")); plt.close(fig)
json.dump(tables, open(os.path.join(TAB, "tables.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
with open(os.path.join(MS, "numbers.tex"), "w") as f:
    f.write("% generated by artifact/code/analyze.py\n")
    for k, v in sorted(num.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(num, indent=1))
