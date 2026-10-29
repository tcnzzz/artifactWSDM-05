"""Held-out test of Proposition 1: realised gain of a judge call as a function of the encoder's prior index.

On test goals, links met by the encoder-only index policy are sampled stratified over prior-index bins, judged by the
LLM, and the realised Lagrangian payoff of the decision taken with the posterior is compared with the decision taken
with the prior, using the TRUE layer of the link (not the model). Writes results/gain_curve.json.
  python eval_gain_curve.py --lam <price> [--per_crawl 25] [--rows index_T]
"""
import os, glob, pickle, json, argparse, time, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common import D, R, Graph, LLM, load_goals
from crawl import MODELS, K, chain_index, Models, value_at
from stats import boot
ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=float, required=True); ap.add_argument("--per_crawl", type=int, default=25)
ap.add_argument("--rows", default="index_T"); ap.add_argument("--model", default="qwen2.5:7b"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--rhos", default="0.05,0.1,0.2,0.5,1.0"); ap.add_argument("--limit_q", type=int, default=100); ap.add_argument("--out_tag", default=""); a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(D, "rows", f"test_{a.rows}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
assert rows, "no rows"
keep = set(list(dict.fromkeys(g["qid"] for g in load_goals("test")))[:a.limit_q]); rows = [r for r in rows if r["gid"].rsplit("_h", 1)[0] in keep]
M = Models(); beta = M.beta; edges = M.post["bin_edges"]; nb = len(edges) - 1; goals = {g["gid"]: g for g in load_goals("test")}
pool = []
for r in rows:
    m = r["in_crawl"]; pr = r["prior"][m]; g = chain_index(pr, beta)[0]
    for j in range(len(g)): pool.append((r["gid"], int(r["par"][m][j]), int(r["tgt"][m][j]), pr[j], int(r["y"][m][j]), g[j]))
gall = np.array([p[5] for p in pool]); pbin = np.clip(np.searchsorted(edges, gall, side="right") - 1, 0, nb - 1)
rng = np.random.default_rng(2); by = {}
for i, p in enumerate(pool): by.setdefault(p[0], []).append(i)
sample = []
for gid, idxs in by.items():
    idxs = np.array(idxs); bins = pbin[idxs]; chosen = []; present = rng.permutation(np.unique(bins)); q = a.per_crawl
    while q > 0 and len(chosen) < len(idxs):
        for b in present:
            cand = [i for i in idxs[bins == b] if i not in chosen]
            if cand and q > 0: chosen.append(int(rng.choice(cand))); q -= 1
    sample += chosen
sample = np.array(sample); print("pool", len(pool), "sample", len(sample), flush=True)
tl = threading.local(); t0 = time.time()
def work(k):
    if not hasattr(tl, "llm"): tl.llm = LLM(a.model); tl.g = Graph()
    gid, par, tgt = pool[sample[k]][:3]; g = goals[gid]; s, ph = tl.llm.inspect(g["qid"], g["question"], tl.g, par, tgt)
    if k % 500 == 0: print(k, round(time.time() - t0), flush=True); tl.llm.flush()
    return s, ph, tl.llm
with ThreadPoolExecutor(a.threads) as ex: res = list(ex.map(work, range(len(sample))))
llms = set(r[2] for r in res); [o.flush() for o in llms]
S = np.array([r[0] for r in res]); PH = np.array([r[1] for r in res]); prior = np.stack([pool[i][3] for i in sample]); y = np.array([pool[i][4] for i in sample])
qid = np.array([pool[i][0].rsplit("_h", 1)[0] for i in sample]); gp = np.array([pool[i][5] for i in sample]); b = np.clip(np.searchsorted(edges, gp, side="right") - 1, 0, nb - 1)
post = M.posterior(prior, S, PH)
from crawl import realised_payoff
def realised(Rc, lam): return realised_payoff(Rc, y, lam, beta)
out = dict(lam=a.lam, n=int(len(y)), bins=[], model=a.model, rhos=[float(x) for x in a.rhos.split(",")],
           new_calls=sum(o.new_calls for o in llms), new_prompt_tokens=sum(o.new_ptok for o in llms))
pay_post, act_post = realised(post, a.lam); pay_prior, act_prior = realised(prior, a.lam); gain = pay_post - pay_prior
T = {rho: M.tables(rho, a.lam) for rho in out["rhos"]}
for j in range(nb):
    m = b == j
    if m.sum() < 10: continue
    mean, lo, hi = boot(gain[m], qid[m], seed=j)
    out["bins"].append(dict(bin=j, g_lo=float(edges[j]), g_hi=float(min(edges[j + 1], 1.0)), g_mean=float(gp[m].mean()), n=int(m.sum()), gain=mean, lo=lo, hi=hi,
                            act_prior=float(act_prior[m].mean()), act_post=float(act_post[m].mean()), base_rate_R1=float((y[m] == 0).mean()), base_rate_RK=float((y[m] < K).mean()),
                            predicted_gain=float(T[out["rhos"][0]]["gain"][j]), predicted_region={str(r): bool(T[r]["region"][j]) for r in out["rhos"]},
                            realised_region={str(r): bool(mean > a.lam * r) for r in out["rhos"]}))
def is_interval(flags):
    idx = [i for i, f in enumerate(flags) if f]; return (not idx) or (idx[-1] - idx[0] + 1 == len(idx))
out["realised_region_is_interval"] = {str(r): is_interval([bb["realised_region"][str(r)] for bb in out["bins"]]) for r in out["rhos"]}
out["predicted_region_is_interval"] = {str(r): is_interval([bb["predicted_region"][str(r)] for bb in out["bins"]]) for r in out["rhos"]}
out["region_width_bins"] = {str(r): dict(realised=sum(bb["realised_region"][str(r)] for bb in out["bins"]), predicted=sum(bb["predicted_region"][str(r)] for bb in out["bins"])) for r in out["rhos"]}
out["peak_bin_realised"] = int(np.argmax([bb["gain"] for bb in out["bins"]])); out["peak_g_mean"] = out["bins"][out["peak_bin_realised"]]["g_mean"]
json.dump(out, open(os.path.join(R, f"gain_curve{a.out_tag}.json"), "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "bins"}, indent=1))
for bb in out["bins"]: print(bb["bin"], round(bb["g_mean"], 4), bb["n"], round(bb["gain"], 4), round(bb["lo"], 4), round(bb["hi"], 4), round(bb["predicted_gain"], 4))
