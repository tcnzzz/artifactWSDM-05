"""Build crawl goals from HotpotQA bridge questions on the Wikipedia intro link graph.

A goal is (question, gold pages P1 -> P2 with a real hyperlink P1 -> P2, seed page). The seed is placed
h in {1,2,3} hops upstream of P1 by a backward random walk and kept only if the exact shortest distances
are dist(seed,P1)=h and dist(seed,P2)=h+1, so target depth is known exactly (controlled factor: seed placement).
Splits: train/cal/val from HotpotQA train, test from HotpotQA dev; no gold page of a test goal appears in another split.
Also writes PageRank (full graph) and BM25 document frequencies.
"""
import os, json, glob, sys
import numpy as np, pandas as pd
from common import Graph, D, G, toks
rng = np.random.default_rng(20261005)
HP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--hotpotqa--hotpot_qa/snapshots/*/distractor"))[0]
SIZES = dict(test=400, train=1200, cal=400, val=150)
g = Graph()

def fwd_dist(seed, targets, maxd):
    """Exact BFS distances from seed to each target, up to maxd (None if farther)."""
    seen = {seed}; cur = [seed]; res = {t: (0 if t == seed else None) for t in targets}
    for d in range(1, maxd + 1):
        nxt = []
        for v in cur:
            for w in g.out(v):
                w = int(w)
                if w not in seen: seen.add(w); nxt.append(w)
        for t in targets:
            if res[t] is None and t in seen: res[t] = d
        cur = nxt
        if all(r is not None for r in res.values()) or len(seen) > 400000: break
    return res

def make_seed(p1, p2, h):
    for _ in range(40):
        path = [p1]; ok = True
        for _s in range(h):
            inn = g.inn(path[0]); inn = inn[(inn != p2) & (inn != p1)]
            if len(inn) == 0: ok = False; break
            x = int(rng.choice(inn))
            if x in path: ok = False; break
            path.insert(0, x)
        if not ok: continue
        seed = path[0]; deg = len(g.out(seed))
        if deg < 3 or deg > 80: continue
        r = fwd_dist(seed, [p1, p2], h + 1)
        if r[p1] == h and r[p2] == h + 1: return path + [p2]
    return None

def candidates(df):
    out = []
    for r in df.itertuples():
        if r.type != "bridge": continue
        ts = list(dict.fromkeys(r.supporting_facts["title"]))
        if len(ts) != 2: continue
        a, b = g.idx(ts[0]), g.idx(ts[1])
        if a is None or b is None: continue
        ab = b in g.out(a); ba = a in g.out(b)
        if ab and not ba: p1, p2 = a, b
        elif ba and not ab: p1, p2 = b, a
        else: continue                       # need exactly one direction so that P1 -> P2 is unambiguous
        out.append(dict(qid=r.id, question=r.question, answer=r.answer, level=r.level, p1=int(p1), p2=int(p2)))
    return out

def build(cands, n, split, used):
    goals = []; nq = 0
    for i in rng.permutation(len(cands)):
        c = cands[i]
        if c["p1"] in used or c["p2"] in used: continue
        paths = [make_seed(c["p1"], c["p2"], h) for h in (1, 2, 3)]
        if any(p is None for p in paths): continue
        used.update([c["p1"], c["p2"]]); nq += 1
        for h, p in zip((1, 2, 3), paths):
            goals.append(dict(c, gid=f"{c['qid']}_h{h}", split=split, h=h, depth=h + 1, seed=p[0], path=p))
        if nq >= n: break
    with open(os.path.join(D, f"goals_{split}.jsonl"), "w") as f:
        for x in goals: f.write(json.dumps(x) + "\n")
    print(split, nq, "questions", len(goals), "goals", flush=True); return nq

if __name__ == "__main__":
    dev = pd.read_parquet(glob.glob(HP + "/validation-*.parquet")[0]); tr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(HP + "/train-*.parquet"))])
    cd = candidates(dev); print("dev bridge candidates with a one-way gold link:", len(cd), flush=True)
    used = set(); stats = dict(dev_candidates=len(cd)); stats["test"] = build(cd, SIZES["test"], "test", used)
    ct = candidates(tr.sample(n=12000, random_state=7)); stats["train_candidates_sampled"] = len(ct)
    ct = [c for c in ct if c["p1"] not in used and c["p2"] not in used]; k = 0
    for split in ("val", "cal", "train"):
        stats[split] = build(ct, SIZES[split], split, used)
    json.dump(stats, open(os.path.join(D, "goals_stats.json"), "w"), indent=1)
    # PageRank on the full graph (power iteration, damping 0.85)
    import scipy.sparse as sp
    n = g.n; deg = np.diff(np.asarray(g.oi)); src = np.repeat(np.arange(n), deg)
    M = sp.csr_matrix((1.0 / deg[src], (np.asarray(g.ox), src)), shape=(n, n)); pr = np.full(n, 1.0 / n); dang = deg == 0
    for it in range(50):
        new = 0.85 * (M @ pr + pr[dang].sum() / n) + 0.15 / n; delta = np.abs(new - pr).sum(); pr = new
        if delta < 1e-9: break
    np.save(os.path.join(G, "pagerank.npy"), pr.astype(np.float32)); print("pagerank iters", it, flush=True)
    # BM25 document frequencies over link sentences of a page sample
    df = {}; N = 0; tot = 0
    for v in rng.choice(n, 200000, replace=False):
        for s in g.page(int(v))[1]:
            t = toks(s); N += 1; tot += len(t)
            for w in set(t): df[w] = df.get(w, 0) + 1
    json.dump(dict(df=df, N=N, avgdl=tot / max(N, 1)), open(os.path.join(G, "bm25_df.json"), "w"))
