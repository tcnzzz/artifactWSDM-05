"""LLM inspection signals on calibration goals, and the posterior model P(layer | encoder prior, LLM signal).

1. Sample links met by the calibrated index policy on calibration goals, stratified over prior-index bins.
2. Query the LLM once per sampled link (cached).
3. Cross-fit by question (5 folds): multinomial logistic regression, then isotonic calibration fitted on the other
   folds' out-of-fold scores. The out-of-fold posteriors feed the inspection tables; final models are refitted on all folds.
  python fit_posterior.py <rows-kind> [--model qwen2.5:7b] [--per_crawl 30] [--tag ""]
"""
import sys, os, glob, pickle, json, argparse, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from common import D, Graph, LLM, load_goals
from crawl import MODELS, K, chain_index, logit, LRWrap, OffsetPosterior
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("--model", default="qwen2.5:7b"); ap.add_argument("--per_crawl", type=int, default=30)
ap.add_argument("--tag", default=""); ap.add_argument("--post", default="offset"); ap.add_argument("--calib_rows", default=""); ap.add_argument("--nbins", type=int, default=20); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(D, "rows", f"cal_{a.kind}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
beta = pickle.load(open(os.path.join(MODELS, "calib.pkl"), "rb"))["beta"]; goals = {g["gid"]: g for g in load_goals("cal")}
rng = np.random.default_rng(1); pool = []
for r in rows:
    m = r["in_crawl"]; pr = r["prior"][m]; g = chain_index(pr, beta)[0]
    for j in range(len(g)): pool.append((r["gid"], int(r["par"][m][j]), int(r["tgt"][m][j]), pr[j], int(r["y"][m][j]), g[j], r["X"][m][j]))
gall = np.array([p[5] for p in pool]); edges = np.unique(np.quantile(gall, np.linspace(0, 1, a.nbins + 1))); edges[0] = 0.0; edges[-1] = np.inf
nb = len(edges) - 1; pbin = np.clip(np.searchsorted(edges, gall, side="right") - 1, 0, nb - 1)
# stratified sample: per crawl, spread the quota evenly over the bins present
by = {}
for i, p in enumerate(pool): by.setdefault(p[0], []).append(i)
sample = []
for gid, idxs in by.items():
    idxs = np.array(idxs); bins = pbin[idxs]; chosen = []
    present = rng.permutation(np.unique(bins)); q = a.per_crawl
    while q > 0 and len(chosen) < len(idxs):
        for b in present:
            cand = [i for i in idxs[bins == b] if i not in chosen]
            if cand and q > 0: chosen.append(int(rng.choice(cand))); q -= 1
    sample += chosen
sample = np.array(sample); cnt_all = np.bincount(pbin, minlength=nb); cnt_s = np.bincount(pbin[sample], minlength=nb); w = (cnt_all / np.maximum(cnt_s, 1))[pbin[sample]]
print("pool", len(pool), "sample", len(sample), "bins", nb, flush=True)
graph = Graph(); llm = LLM(a.model); t0 = time.time(); S = np.zeros(len(sample)); PH = np.zeros(len(sample))
import threading
# sequential access to sqlite, parallel HTTP: use one LLM object per thread
tl = threading.local()
def work(k):
    if not hasattr(tl, "llm"): tl.llm = LLM(a.model); tl.g = Graph()
    gid, par, tgt = pool[sample[k]][:3]; g = goals[gid]; s, ph = tl.llm.inspect(g["qid"], g["question"], tl.g, par, tgt)
    if k % 500 == 0: print(k, round(time.time() - t0), flush=True); tl.llm.flush()
    return s, ph, tl.llm
with ThreadPoolExecutor(a.threads) as ex: res = list(ex.map(work, range(len(sample))))
for o in set(r[2] for r in res): o.flush()
S = np.array([r[0] for r in res]); PH = np.array([r[1] for r in res])
prior = np.stack([pool[i][3] for i in sample]); y = np.array([pool[i][4] for i in sample]); qid = np.array([pool[i][0].rsplit("_h", 1)[0] for i in sample])
uq = np.unique(qid); fold = {q: i % 5 for i, q in enumerate(rng.permutation(uq))}; fo = np.array([fold[q] for q in qid])
prior_infold = prior.copy()
if a.calib_rows:
    # Cross-fitted prior (round 1): the scorer's isotonic calibrators are refitted by question fold on the calibration-crawl rows
    # they were originally fitted on, and each sampled link gets the prior from the calibrators that did not see its question.
    import lightgbm as lgb
    calp = pickle.load(open(os.path.join(MODELS, "calib.pkl"), "rb")); dense = bool(calp.get("dense"))
    def raw_cum(X):
        if dense: return np.tile((0.5 * (X[:, 0] + X[:, 1]))[:, None], (1, K))
        return np.cumsum(lgb.Booster(model_file=os.path.join(MODELS, "encoder.txt")).predict(X)[:, :K], 1)
    crow = []
    for f_ in sorted(glob.glob(os.path.join(D, "rows", f"cal_{a.calib_rows}_s*.pkl"))): crow += pickle.load(open(f_, "rb"))
    CX = np.vstack([r["X"][r["in_crawl"]] for r in crow]); Cy = np.concatenate([r["y"][r["in_crawl"]] for r in crow])
    Cq = np.concatenate([[r["gid"].rsplit("_h", 1)[0]] * int(r["in_crawl"].sum()) for r in crow]); Cf = np.array([fold.get(q, hash(q) % 5) for q in Cq])
    Ccum = raw_cum(CX); Scum = raw_cum(np.stack([pool[i][6] for i in sample])); prior = np.zeros_like(prior_infold)
    for f_ in range(5):
        isos = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(Ccum[Cf != f_, k], (Cy[Cf != f_] <= k).astype(float)) for k in range(K)]
        prior[fo == f_] = np.stack([isos[k].predict(Scum[fo == f_, k]) for k in range(K)], 1)
    prior = np.maximum.accumulate(np.clip(prior, 1e-5, 1 - 1e-5), 1)
F = np.column_stack([logit(prior), S, PH, S * S])
def fit_lr(F_, y_, w_): return LogisticRegression(C=1.0, max_iter=2000).fit(F_, y_, sample_weight=w_)
def fit_iso(cum, y_, w_): return [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(cum[:, k], (y_ <= k).astype(float), sample_weight=w_) for k in range(K)]
def full_classes(lr, P):                          # make sure all K+1 classes have a column
    out = np.zeros((len(P), K + 1)); out[:, lr.classes_] = P; return out
oof = np.zeros((len(y), K + 1))
for f in range(5):
    lr = fit_lr(F[fo != f], y[fo != f], w[fo != f]); oof[fo == f] = full_classes(lr, lr.predict_proba(F[fo == f]))
cum = np.cumsum(oof[:, :K], 1); post_oof = np.zeros((len(y), K))
for f in range(5):
    iso = fit_iso(cum[fo != f], y[fo != f], w[fo != f]); post_oof[fo == f] = np.stack([iso[k].predict(cum[fo == f, k]) for k in range(K)], 1)
post_oof = np.maximum.accumulate(np.clip(post_oof, 1e-5, 1 - 1e-5), 1)
if a.post == "offset":                            # Bayes-form posterior (default); the weighted multinomial fit above is kept as kind v0
    def fit_off(idx):
        return OffsetPosterior([LogisticRegression(C=10.0, max_iter=5000).fit(OffsetPosterior.feats(prior[idx, k], S[idx], PH[idx]), (y[idx] <= k).astype(int)) for k in range(K)])
    for f in range(5): post_oof[fo == f] = fit_off(fo != f).predict(prior[fo == f], S[fo == f], PH[fo == f])
    final = fit_off(np.ones(len(y), bool)); lr = None; iso_post = None
else: final = None; lr = LRWrap(fit_lr(F, y, w)); iso_post = fit_iso(cum, y, w)
iso_llm = fit_iso(np.tile(S[:, None], (1, K)), y, w)
pickle.dump(dict(kind=a.post, model=final, lr=lr, iso_post=iso_post, iso_llm=iso_llm, bin_edges=edges, cal=dict(prior=prior, prior_infold=prior_infold, post_oof=post_oof, y=y, s=S, phi=PH, w=w, qid=qid), crossfit_prior=bool(a.calib_rows)),
            open(os.path.join(MODELS, f"posterior{a.tag}.pkl"), "wb"))
def nll(Rk, yk, w_): Rk = np.clip(Rk, 1e-5, 1 - 1e-5); return float(-(w_ * (yk * np.log(Rk) + (1 - yk) * np.log(1 - Rk))).sum() / w_.sum())
def auc(sc, yk):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(yk, sc)) if 0 < yk.sum() < len(yk) else None
from crawl import realised_payoff
beta_ = pickle.load(open(os.path.join(MODELS, "calib.pkl"), "rb"))["beta"]
info = dict(kind=a.post, rows=a.kind, model=a.model, n=int(len(y)), new_calls=sum(o.new_calls for o in set(r[2] for r in res)), new_prompt_tokens=sum(o.new_ptok for o in set(r[2] for r in res)))
for k in range(K):
    yk = (y <= k).astype(float)
    info[f"R{k+1}"] = dict(base=float((w * yk).sum() / w.sum()), nll_prior=nll(prior[:, k], yk, w), nll_post_oof=nll(post_oof[:, k], yk, w),
                           auc_prior=auc(prior[:, k], yk), auc_llm=auc(S, yk), auc_post_oof=auc(post_oof[:, k], yk))
lam_ = json.load(open(os.path.join(MODELS, "lambda.json")))["25"]["lam"] if os.path.exists(os.path.join(MODELS, "lambda.json")) else 0.015
gq_ = chain_index(prior, beta_)[0]; gb = np.clip(np.searchsorted(edges, gq_, side="right") - 1, 0, nb - 1)
dg = realised_payoff(post_oof, y, lam_, beta_)[0] - realised_payoff(prior, y, lam_, beta_)[0]
info["realised_gain_over_lam_by_bin"] = [round(float(dg[gb == j].mean() / lam_), 4) for j in range(nb)]; info["lam"] = lam_
json.dump(info, open(os.path.join(MODELS, f"posterior_info{a.tag}.json"), "w"), indent=1); print(json.dumps(info, indent=1))
