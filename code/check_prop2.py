"""Numeric check of Proposition 2 (max-ratio index on forests) by brute force.

Forest of pages; node v has fetch cost c_v > 0 and value p_v >= 0 (probability that v is a
target; recall is additive). A set S is feasible if it is ancestor-closed. Index
    gamma(v) = max over nonempty subtrees T rooted at v of p(T)/c(T).
Index policy: always fetch the available node with the largest index.
Checked on random forests: for every lam, the policy fetches S_lam = {v: gamma(u)>lam on the
whole root-v path} before any other node, S_lam maximises p(S)-lam*c(S), and at each breakpoint
budget c(S_lam) the prefix is optimal among all feasible sets within that budget.
Also evaluates the three-node gateway instance. Writes results/theory/prop2_check.json.
"""
import json, os, itertools, numpy as np
rng = np.random.default_rng(20261005)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "theory"); os.makedirs(OUT, exist_ok=True)

def index(par, p, c):
    n = len(par); ch = [[] for _ in range(n)]
    for v, u in enumerate(par):
        if u >= 0: ch[u].append(v)
    def g(v, lam): return p[v] - lam * c[v] + sum(max(0.0, g(w, lam)) for w in ch[v])
    gam = np.zeros(n)
    for v in range(n):
        lo, hi = 0.0, 1e6                      # g(v,lam) is decreasing in lam; root by bisection
        for _ in range(200):
            mid = (lo + hi) / 2
            if g(v, mid) > 0: lo = mid
            else: hi = mid
        gam[v] = lo
    return gam, ch

def closed_sets(par):
    n = len(par)
    for mask in range(1 << n):
        if all((mask >> par[v]) & 1 for v in range(n) if (mask >> v) & 1 and par[v] >= 0): yield mask

def run_policy(par, key):
    n = len(par); done = []; avail = {v for v in range(n) if par[v] < 0}
    while avail:
        v = max(avail, key=lambda x: (key[x], -x)); avail.remove(v); done.append(v)
        avail |= {w for w in range(n) if par[w] == v}
    return done

bad = dict(order=0, lagr=0, budget=0); n_inst = 1500; nb = 0
for _ in range(n_inst):
    n = int(rng.integers(3, 11))
    par = [-1 if (v == 0 or rng.random() < 0.2) else int(rng.integers(0, v)) for v in range(n)]
    p = rng.random(n) * (rng.random(n) < 0.6); c = rng.uniform(0.5, 2.0, n)
    gam, ch = index(par, p, c); order = run_policy(par, gam)
    sets = list(closed_sets(par)); val = lambda m, w: sum(w[v] for v in range(n) if (m >> v) & 1)
    def path(v):
        out = []
        while v >= 0: out.append(v); v = par[v]
        return out
    for lam in list(np.unique(gam)) + [0.0]:
        lam2 = max(lam - 1e-6, 0.0)
        S = {v for v in range(n) if all(gam[u] > lam2 for u in path(v))}
        k = len(S); nb += 1
        if set(order[:k]) != S: bad["order"] += 1
        mS = sum(1 << v for v in S)
        best = max(val(m, p) - lam2 * val(m, c) for m in sets)
        if val(mS, p) - lam2 * val(mS, c) < best - 1e-6: bad["lagr"] += 1
        B = val(mS, c); bestB = max(val(m, p) for m in sets if val(m, c) <= B + 1e-12)
        if val(mS, p) < bestB - 1e-6: bad["budget"] += 1

# three-node gateway instance: decoy d (value eps), gateway g (value 0) -> target t (value 1)
gw = {}
for eps in [0.3, 0.1, 0.01, 0.001]:
    par = [-1, -1, 1]; p = np.array([eps, 0.0, 1.0]); c = np.ones(3)
    gam, _ = index(par, p, c)
    greedy = run_policy(par, p / c)[:2]; idx = run_policy(par, gam)[:2]
    gw[str(eps)] = dict(index_gateway=float(gam[1]), greedy_value=float(p[greedy].sum()), index_value=float(p[idx].sum()),
                        ratio=float(p[idx].sum() / p[greedy].sum()))
out = dict(n_instances=n_inst, n_breakpoints=nb, violations=bad, gateway_budget2=gw)
json.dump(out, open(os.path.join(OUT, "prop2_check.json"), "w"), indent=1); print(json.dumps(out, indent=1))
