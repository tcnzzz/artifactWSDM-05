"""Budgeted crawl simulator and all frontier policies.

Cost model: a fetch costs 1, an LLM inspection of one link costs rho. A crawl stops when the budget is spent.
Every policy sees only what a crawler sees: fetched pages and the anchor text / sentence of their links.
"""
import os, json, math, pickle
import numpy as np
from common import Graph, Embedder, LLM, BM25, link_features, toks, D, G, R, FEATS
MODELS = os.environ.get("SF_MODELS") or os.path.join(D, "models"); os.makedirs(MODELS, exist_ok=True)
K = 3                                   # horizon: a link is priced for targets up to K fetches away
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def chain_index(Rc, beta):
    """Max-ratio index of a link from its cumulative layer probabilities Rc[:, k-1] = P(gold within k fetches).
    Commit depth k collects Rc[:,k-1]; the expected cost is sum_j pi_j c(j) + (1-R_k) c(k), c(j) = 1 + beta (j-1)."""
    Rc = np.atleast_2d(Rc); pi = np.diff(np.concatenate([np.zeros((len(Rc), 1)), Rc], 1), axis=1)
    c = 1 + beta * np.arange(K); cumc = np.cumsum(pi * c, 1); C = cumc + (1 - Rc) * c
    ratio = Rc / C; k = ratio.argmax(1); a = np.arange(len(Rc))
    return ratio[a, k], Rc[a, k], C[a, k], Rc, C
def value_at(Rc, C, lam):
    """Lagrangian value of a link at budget price lam: max(0, max_k R_k - lam C_k)."""
    return np.maximum((Rc - lam * C).max(1), 0.0)

class Identity:
    def predict(self, x): return np.asarray(x, float)
class LRWrap:
    """Logistic-regression wrapper that always returns K+1 class columns."""
    def __init__(self, lr): self.lr = lr
    def predict_proba(self, F):
        out = np.zeros((len(F), K + 1)); out[:, self.lr.classes_] = self.lr.predict_proba(F); return out

class OffsetPosterior:
    """Bayes-form posterior: for each k, logit P(L <= k | prior, signal) = a_k logit(R_k) + h_k(signal), fitted by logistic
    regression without sample weights (selection depends only on the prior, a covariate). With an uninformative signal it
    reduces to a recalibration of the prior, so the decision value cannot be degraded by a badly weighted fit."""
    def __init__(self, models): self.m = models
    @staticmethod
    def feats(Rk, s, phi): z = logit(Rk); return np.column_stack([z, s, phi, s * z, s * s])
    def predict(self, Rc, s, phi):
        P = np.stack([self.m[k].predict_proba(self.feats(Rc[:, k], s, phi))[:, 1] for k in range(K)], 1)
        return np.maximum.accumulate(np.clip(P, 1e-5, 1 - 1e-5), 1)

class Models:
    """Encoder head, calibrators, posterior model and the inspection tables (all fitted on train/cal goals only)."""
    def __init__(self, tag=""):
        import lightgbm as lgb
        self.cal = pickle.load(open(os.path.join(MODELS, "calib.pkl"), "rb"))        # iso_prior[k], beta, dense flag
        self.dense = bool(self.cal.get("dense"))                                      # weak regime: zero-shot dense score, calibrated
        self.enc = None if self.dense else lgb.Booster(model_file=os.path.join(MODELS, "encoder.txt"))
        self.beta = self.cal["beta"]; self.post = None
        pb = os.path.join(MODELS, "encoder_bin.txt"); self.enc_bin = lgb.Booster(model_file=pb) if os.path.exists(pb) else None
        p = os.path.join(MODELS, f"posterior{tag}.pkl")
        if os.path.exists(p): self.post = pickle.load(open(p, "rb"))                 # lr, iso_post[k], iso_llm[k], cal arrays
    def prior(self, X):
        if self.dense: cum = np.tile((0.5 * (X[:, 0] + X[:, 1]))[:, None], (1, K))
        else: raw = self.enc.predict(X); cum = np.cumsum(raw[:, :K], 1)
        Rc = np.stack([self.cal["iso_prior"][k].predict(cum[:, k]) for k in range(K)], 1)
        return np.maximum.accumulate(np.clip(Rc, 1e-5, 1 - 1e-5), 1)
    def post_feats(self, Rc, s, phi): return np.column_stack([logit(Rc), s, phi, s * s])
    def posterior(self, Rc, s, phi):
        if self.post.get("kind") == "offset": return self.post["model"].predict(Rc, np.asarray(s, float), np.asarray(phi, float))
        raw = self.post["lr"].predict_proba(self.post_feats(Rc, s, phi)); cum = np.cumsum(raw[:, :K], 1)
        Rq = np.stack([self.post["iso_post"][k].predict(cum[:, k]) for k in range(K)], 1)
        return np.maximum.accumulate(np.clip(Rq, 1e-5, 1 - 1e-5), 1)
    def llm_only(self, s):
        Rq = np.stack([self.post["iso_llm"][k].predict(s) for k in range(K)], 1)
        return np.maximum.accumulate(np.clip(Rq, 1e-5, 1 - 1e-5), 1)
    def tables(self, rho, lam, nbins=20, conservative=False):
        """Per prior-index bin of calibration links: realised gain of a judge call at price lam (payoff of the decision taken
        with the out-of-fold posterior minus payoff of the decision taken with the prior, both scored with the TRUE layer),
        and the two-stage (inspect first) index of the bin."""
        c = self.post["cal"]; gp = chain_index(c["prior"], self.beta)[0]; gq, Rq, Cq, _, _ = chain_index(c["post_oof"], self.beta)
        edges = self.post["bin_edges"]; b = np.clip(np.searchsorted(edges, gp, side="right") - 1, 0, len(edges) - 2); nb = len(edges) - 1
        pay_q, act_q = realised_payoff(c["post_oof"], c["y"], lam, self.beta); pay_p, _ = realised_payoff(c["prior"], c["y"], lam, self.beta)
        g = pay_q - pay_p; gain = np.zeros(nb); iidx = np.zeros(nb); pfetch = np.zeros(nb); cnt = np.zeros(nb, int)
        for j in range(nb):
            m = b == j; cnt[j] = m.sum()
            if m.sum() < 5: continue
            gain[j] = g[m].mean(); o = np.argsort(-gq[m]); r = np.cumsum(Rq[m][o]); cc = m.sum() * rho + np.cumsum(Cq[m][o])
            iidx[j] = (r / cc).max(); pfetch[j] = act_q[m].mean()
        region = gain > lam * rho; lcb = None
        if conservative:
            # Conservative region: a bin escalates only if the one-sided 95% lower bound of its gain (bootstrap over
            # calibration questions, 1000 resamples, seed 0) exceeds the price of a call, and, as Proposition 1 requires,
            # only the contiguous run of such bins that contains the bin of largest estimated gain is kept.
            uq, inv = np.unique(c["qid"], return_inverse=True); S = np.zeros((len(uq), nb)); N = np.zeros((len(uq), nb))
            np.add.at(S, (inv, b), g); np.add.at(N, (inv, b), 1); idx = np.random.default_rng(0).integers(0, len(uq), (1000, len(uq)))
            lcb = np.percentile(S[idx].sum(1) / np.maximum(N[idx].sum(1), 1), 5, axis=0); ok = lcb > lam * rho; pk = int(np.argmax(gain)); region = np.zeros(nb, bool)
            if conservative == "all": region = ok.copy()       # final rule (round 1): every bin whose lower bound passes, no contiguity step
            elif ok[pk]:                                       # round-0 exploration rule: run of passing bins around the point-estimate peak
                lo = pk; hi = pk
                while lo > 0 and ok[lo - 1]: lo -= 1
                while hi < nb - 1 and ok[hi + 1]: hi += 1
                region[lo:hi + 1] = True
        return dict(edges=edges, gain=gain, region=region, insp_index=iidx, pfetch=pfetch, lam=lam, rho=rho, count=cnt, lcb=lcb)

def realised_payoff(Rc, y, lam, beta):
    """Lagrangian payoff of acting on calibrated layer probabilities Rc at price lam, scored with the true layer y:
    act iff max_k R_k - lam C_k > 0 at the best k; reward 1[L <= k], cost c(min(L, k)), L = true fetches to the target."""
    _, _, _, Rcc, C = chain_index(Rc, beta); V = Rcc - lam * C; k = V.argmax(1) + 1; act = V.max(1) > 0
    c = 1 + beta * np.arange(K); L = np.where(y < K, y + 1, 10**6)
    return np.where(act, (L <= k).astype(float) - lam * c[np.minimum(L, k) - 1], 0.0), act

class Crawl:
    def __init__(self, goal, graph, emb, models, rho, budget, llm=None, extras=None):
        self.g = goal; self.G = graph; self.E = emb; self.M = models; self.rho = rho; self.B = budget; self.llm = llm; self.X = extras or {}
        self.q = goal["question"]; self.qv = emb.query(self.q); self.qt = toks(self.q); self.gold = {goal["p1"], goal["p2"]}
        self.dist = graph.layers(self.gold, depth=K - 1)
        self.cost = 0.0; self.nllm = 0; self.events = []; self.seen = {goal["seed"]}; self.fetched = []
        n0 = 0; self.tgt = np.zeros(n0, np.int64); self.par = np.zeros(n0, np.int64); self.dep = np.zeros(n0, np.int32); self.alive = np.zeros(n0, bool)
        self.ca = np.zeros(n0, np.float32); self.cs = np.zeros(n0, np.float32); self.Rp = np.zeros((n0, K)); self.Rc = np.zeros((n0, K)); self.insp = np.zeros(n0, bool)
        self.sig = np.full(n0, np.nan); self.bm = np.zeros(n0, np.float32); self.av = np.zeros((n0, 384), np.float32); self.tv = []
        self.gidx = np.zeros(n0); self.gprior = np.zeros(n0); self.keepX = bool(self.X.get("keepX")); self.Xs = []; self.pbin = np.zeros(n0)
        self._fetch_page(goal["seed"], 0, 0.0, 0.0, count=False)
    def _fetch_page(self, page, depth, pca, pcs, count=True, g=0.0):
        if count:
            self.cost += 1.0; self.events.append(("F", round(self.cost, 4), int(page in self.gold), int(self.dist[page]), depth, round(float(g), 5), int(page))); self.fetched.append(page)
        title, sents, links = self.G.page(page); tv, sv, av = self.E.page(self.G, page); self.tv.append(tv); self.new = np.zeros(0, np.int64)
        keep = [i for i, l in enumerate(links) if l[2] not in self.seen]
        if not keep: return
        X = link_features(self.G, self.E, self.qv, self.qt, page, depth, pca, pcs)[keep]; n = len(keep)
        for i in keep: self.seen.add(links[i][2])
        Rp = self.M.prior(X); g = chain_index(Rp, self.M.beta)[0]
        if self.keepX: self.Xs.append(X)
        self.pbin = np.r_[self.pbin, self.M.enc_bin.predict(X) if getattr(self.M, "enc_bin", None) is not None else np.zeros(n)]
        bm = [self.X["bm25"].score(self.qt, links[i][0] + " " + sents[links[i][1]]) for i in keep] if "bm25" in self.X else np.zeros(n)
        self.tgt = np.r_[self.tgt, [links[i][2] for i in keep]]; self.par = np.r_[self.par, [page] * n]; self.dep = np.r_[self.dep, [depth + 1] * n]
        self.alive = np.r_[self.alive, np.ones(n, bool)]; self.ca = np.r_[self.ca, X[:, 0]]; self.cs = np.r_[self.cs, X[:, 1]]
        self.Rp = np.vstack([self.Rp, Rp]); self.Rc = np.vstack([self.Rc, Rp]); self.insp = np.r_[self.insp, np.zeros(n, bool)]; self.sig = np.r_[self.sig, np.full(n, np.nan)]
        self.bm = np.r_[self.bm, bm]; self.av = np.vstack([self.av, av[keep]]); self.gidx = np.r_[self.gidx, g]; self.gprior = np.r_[self.gprior, g]
        self.new = np.arange(len(self.tgt) - n, len(self.tgt))
    def can_fetch(self): return self.cost + 1.0 <= self.B + 1e-9 and self.alive.any()
    def can_inspect(self): return self.cost + self.rho + 1.0 <= self.B + 1e-9
    def fetch(self, i):
        self.alive[i] = False; self._fetch_page(int(self.tgt[i]), int(self.dep[i]), float(self.ca[i]), float(self.cs[i]), g=self.gidx[i])
    def inspect(self, i, mode="post"):
        s, phi = self.llm.inspect(self.g["qid"], self.q, self.G, int(self.par[i]), int(self.tgt[i]))
        self.cost += self.rho; self.nllm += 1; self.insp[i] = True; self.sig[i] = s
        self.Rc[i] = self.M.posterior(self.Rp[i:i + 1], np.array([s]), np.array([phi]))[0] if mode == "post" else self.M.llm_only(np.array([s]))[0]
        self.gidx[i] = chain_index(self.Rc[i:i + 1], self.M.beta)[0][0]
        self.events.append(("I", round(self.cost, 4), int(self.tgt[i] in self.gold), int(self.dist[self.tgt[i]]), int(self.dep[i]), round(float(self.gprior[i]), 5), round(float(self.gidx[i]), 5), int(self.tgt[i])))
    def top(self, key):
        k = np.where(self.alive, key, -np.inf); return int(k.argmax())

def run(policy, c, P):
    """Runs one crawl with the named policy. P holds policy parameters."""
    import zlib; rng = np.random.default_rng([P.get("seed", 0), zlib.crc32(c.g["gid"].encode())])
    n_arr = lambda: np.arange(len(c.tgt))
    if policy == "llm_only":                                  # the LLM scores every link when it enters the frontier
        pending = list(c.new)
        while c.can_fetch():
            while pending and c.can_inspect(): c.inspect(pending.pop(0), mode="llm")
            key = np.where(c.insp, c.gidx, -1.0) - 1e-9 * n_arr(); c.fetch(c.top(key)); pending += list(c.new)
        return
    if policy == "thompson":
        kap = P.get("ts_kappa", 3.0)
    if policy in ("ours", "ours_online", "band"): T = P["tables"]
    last_new = -1
    while c.can_fetch():
        n = len(c.tgt)
        if policy == "bfs": i = c.top(-n_arr().astype(float)); c.fetch(i)
        elif policy == "random": i = c.top(rng.random(n)); c.fetch(i)
        elif policy == "behav":                                 # data-collection policy: epsilon-greedy on the dense score
            i = c.top(rng.random(n)) if rng.random() < 0.3 else c.top(0.5 * (c.ca + c.cs)); c.fetch(i)
        elif policy == "pagerank": i = c.top(c.X["pr"][c.tgt]); c.fetch(i)
        elif policy == "bm25": i = c.top(c.bm - 1e-6 * n_arr()); c.fetch(i)
        elif policy == "dense": i = c.top(0.5 * (c.ca + c.cs)); c.fetch(i)
        elif policy == "coverage":                              # greedy marginal coverage: relevance x novelty w.r.t. fetched pages
            nov = 1.0 - np.clip((c.av @ np.stack(c.tv).T).max(1), 0, 1); i = c.top(0.5 * (c.ca + c.cs) * nov); c.fetch(i)
        elif policy == "greedy": i = c.top(c.Rp[:, 0]); c.fetch(i)       # myopic: immediate target probability of the multi-class head
        elif policy == "greedy_bin": i = c.top(c.pbin); c.fetch(i)      # strictly myopic: binary head trained on the immediate label only
        elif policy == "reach": i = c.top(c.Rp[:, K - 1]); c.fetch(i)    # reach probability, no cost pricing
        elif policy == "index": i = c.top(c.gprior); c.fetch(i)          # max-ratio index, encoder only
        elif policy == "always":                                # inspect every link before it is fetched
            i = c.top(c.gidx)
            if not c.insp[i] and c.can_inspect(): c.inspect(i)
            else: c.fetch(i)
        elif policy == "topm":                                  # top-m by score plus the most uncertain links go to the LLM
            m, mu = P.get("m", 3), P.get("mu", 1); cand = list(np.argsort(-np.where(c.alive, c.gidx, -np.inf))[:m])
            unc = np.where(c.alive & ~c.insp, -np.abs(c.Rp[:, K - 1] - 0.5), -np.inf); cand += list(np.argsort(-unc)[:mu])
            did = False
            for j in cand:
                if c.alive[j] and not c.insp[j] and c.can_inspect(): c.inspect(j); did = True
            c.fetch(c.top(c.gidx))
        elif policy == "thompson":
            # Thompson sampling in the spirit of Mango: arm = frontier link, prior Beta(1 + kappa r, 1 + kappa (1 - r)) with r the
            # min-max normalised encoder index, an LLM judgement as fractional reward (alpha += s, beta += 1 - s); the sampled
            # argmax is judged by the LLM if it has not been judged yet, otherwise fetched.
            al = c.alive; lo, hi = c.gprior[al].min(), c.gprior[al].max(); r = (c.gprior - lo) / (hi - lo + 1e-12); r = np.clip(r, 0, 1)
            sg = np.where(c.insp, np.nan_to_num(c.sig), 0.0); A = 1 + kap * r + np.where(c.insp, sg, 0.0); Bt = 1 + kap * (1 - r) + np.where(c.insp, 1 - sg, 0.0)
            i = c.top(rng.beta(A, Bt))
            if not c.insp[i] and c.can_inspect(): c.inspect(i)
            else: c.fetch(i)
        elif policy == "band":                                  # tuned middle-band cascade: judge iff prior index in [a lam, b lam]
            esc = (c.gprior >= P["a"] * P["lam"]) & (c.gprior <= P["b"] * P["lam"]) & ~c.insp
            b = np.clip(np.searchsorted(T["edges"], c.gprior, side="right") - 1, 0, len(T["edges"]) - 2)
            key = np.where(esc, np.maximum(c.gidx, T["insp_index"][b]), c.gidx); i = c.top(key)
            if esc[i] and c.can_inspect(): c.inspect(i)
            else: c.fetch(i)
        elif policy in ("ours", "ours_online"):
            if policy == "ours_online":                         # price = index of the marginal link the remaining budget affords
                rem = int(c.B - c.cost); live = np.sort(c.gidx[c.alive])[::-1]; lam = live[min(rem, len(live)) - 1] if len(live) else 0.0
                T = P["models"].tables(c.rho, float(lam)) if abs(lam - P.get("_lam", -1)) > 0.15 * max(lam, 1e-6) else P["_T"]; P["_lam"] = T["lam"]; P["_T"] = T
            b = np.clip(np.searchsorted(T["edges"], c.gprior, side="right") - 1, 0, len(T["edges"]) - 2)
            esc = T["region"][b] & ~c.insp; key = np.where(esc, np.maximum(c.gidx, T["insp_index"][b]), c.gidx); i = c.top(key)
            if esc[i] and c.can_inspect(): c.inspect(i)
            else: c.fetch(i)
        else: raise ValueError(policy)

def summarize(c, budgets):
    ev = c.events; out = dict(gid=c.g["gid"], qid=c.g["qid"], depth=c.g["depth"], n_llm=c.nllm, cost=round(c.cost, 3), n_fetch=len(c.fetched))
    for B in budgets:
        f = [e for e in ev if e[0] == "F" and e[1] <= B + 1e-9]; hit = {}
        out[f"gold@{B}"] = sum(e[2] for e in f); out[f"llm@{B}"] = sum(1 for e in ev if e[0] == "I" and e[1] <= B + 1e-9); out[f"fetch@{B}"] = len(f)
    out["p1_found"] = int(c.g["p1"] in c.fetched); out["p2_found"] = int(c.g["p2"] in c.fetched)
    out["events"] = [list(e) for e in ev]
    return out

class DummyModels:
    """Stand-in before the encoder exists (data collection with the behaviour policy)."""
    beta = 1.0; post = None
    def prior(self, X): return np.tile(np.array([[0.01, 0.02, 0.03]]), (len(X), 1))

def link_rows(c, force_path=True):
    """All links seen in a crawl with their features and true layer (0 = gold, 1, 2, 3 = farther). Optionally also the
    links of the pages on the seed-to-gold path (training positives; never used for calibration or evaluation)."""
    n_crawl = len(c.tgt)
    if force_path:
        for d, p in enumerate(c.g["path"][1:], start=1):
            if p not in c.fetched: c._fetch_page(p, d, 0.0, 0.0, count=False)
    X = np.vstack(c.Xs) if c.Xs else np.zeros((0, len(FEATS)), np.float32)
    return dict(X=X, y=np.minimum(c.dist[c.tgt], K).astype(np.int8), tgt=c.tgt.copy(), par=c.par.copy(), in_crawl=np.arange(len(c.tgt)) < n_crawl,
                fetched_dist=np.array([int(c.dist[p]) for p in c.fetched]))
