"""Diagnostic: how often does the chain index rank a different top link than the myopic score, and what do gateway links look like?
Uses the links met by the encoder-only index policy on test goals (rows test_index_T). Writes results/tables/diag_index.json."""
import os, glob, pickle, json, numpy as np
from common import D, R
from crawl import MODELS, Models, chain_index, K
rows = []
for f in sorted(glob.glob(os.path.join(D, "rows", "test_index_T_s*.pkl"))): rows += pickle.load(open(f, "rb"))
M = Models(); beta = M.beta; out = dict(beta=beta); agree = []; spear = []; gate = dict(n=0, idx_rank_better=0, pi1_rank=[], idx_rank=[])
import scipy.stats as st
for r in rows:
    m = r["in_crawl"]; pr = r["prior"][m]; y = r["y"][m]; par = r["par"][m]
    if len(y) < 5: continue
    g = chain_index(pr, beta)[0]; p1 = pr[:, 0]
    agree.append(int(np.argmax(g) == np.argmax(p1))); spear.append(st.spearmanr(g, p1).correlation)
    # per page: rank of true progress links (layer 1 or 2, i.e. gateways) under the two scores among all links seen so far
    for j in np.where((y == 1) | (y == 2))[0]:
        gate["n"] += 1; rp = int((p1 > p1[j]).sum()); ri = int((g > g[j]).sum()); gate["pi1_rank"].append(rp); gate["idx_rank"].append(ri); gate["idx_rank_better"] += int(ri < rp)
out["frac_same_top_link"] = float(np.mean(agree)); out["spearman_index_vs_pi1_mean"] = float(np.nanmean(spear)); out["n_crawls"] = len(agree)
out["gateway_links"] = dict(n=gate["n"], mean_rank_pi1=float(np.mean(gate["pi1_rank"])), mean_rank_index=float(np.mean(gate["idx_rank"])), frac_index_rank_better=gate["idx_rank_better"] / max(gate["n"], 1))
# how much of the index comes from deeper layers: fraction of links whose argmax commit depth k > 1
Rp = np.vstack([r["prior"][r["in_crawl"]] for r in rows]); ratio, Rs, Cs, Rc, C = chain_index(Rp, beta); k = (Rc / C).argmax(1)
out["frac_links_commit_depth"] = {str(i + 1): float((k == i).mean()) for i in range(K)}; out["n_links"] = int(len(k))
os.makedirs(os.path.join(R, "tables"), exist_ok=True); json.dump(out, open(os.path.join(R, "tables", "diag_index.json"), "w"), indent=1); print(json.dumps(out, indent=1))
