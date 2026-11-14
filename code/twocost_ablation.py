"""Offline ablation of TwoCost variants (H1 difficulty recalibration, H2 confidence/shrinkage,
H3 binning/smoothing) on cached calibration links.

Why offline: the judge backend is not available on this machine, so no new (!) pointwise judge
call can be made.  The calibration posterior file already stores, for every sampled calibration
link, the out-of-fold posterior, the judge signal, the true layer and the question id.  That is
enough to compare decision rules honestly without touching a crawl.

Protocol (pre-declared before looking at the numbers)
-----------------------------------------------------
* Split the 400 calibration questions 50/50 into a tuning half and a held-out half (seed 20261113).
* Build each variant's inspection table on the tuning half only.
* Select the variant that maximises net Lagrangian gain per link on the tuning half, subject to
  (i) at least one and at most nb-1 escalating bins, (ii) inspecting at most 50% of links.
  If no variant beats the deployed rule (full calibration, 20 bins, 95% LCB, all passing bins),
  the deployed rule is frozen and the result is a null.
* Report the selected variant on the held-out half, paired by question against the deployed rule,
  with a cluster bootstrap over questions.
* H1 is supported only if the hard-calibrated table beats the deployed table on hard held-out
  links.  H2 is supported only if a lower confidence level or shrinkage beats the deployed rule
  on held-out links.  H3 is supported only if a different bin count or smoothing does.  If none
  wins, the paper says the ablation found no single mechanism and reports the boundary condition.

Outputs results/tables/twocost_ablation.json.  Writes nothing else.
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import D, R, load_goals
from crawl import K, chain_index, OffsetPosterior, realised_payoff
from stats import load

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

SEED_SPLIT = 20261113
CONF_LEVELS = (0.50, 0.80, 0.90, 0.95)
BIN_LEVELS = (10, 20, 40)
RHOS = (0.1, 1.0)
NBOOT_TABLE = 500
NBOOT_FINAL = 4000


# ----------------------------------------------------------------------------- basic fitting
def raw_cum(X, enc, dense=False):
    if dense:
        return np.tile((0.5 * (X[:, 0] + X[:, 1]))[:, None], (1, K))
    return np.cumsum(enc.predict(X)[:, :K], 1)


def fit_iso(cum, y, w=None):
    iso = []
    for k in range(K):
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(cum[:, k], (y <= k).astype(float), sample_weight=w)
        iso.append(ir)
    return iso


def apply_iso(iso, cum):
    Rc = np.stack([iso[k].predict(cum[:, k]) for k in range(K)], 1)
    return np.maximum.accumulate(np.clip(Rc, 1e-5, 1 - 1e-5), 1)


class WeightedOffset:
    """OffsetPosterior with optional sample weights (the deployed class has no weights)."""

    def __init__(self, models):
        self.m = models

    @staticmethod
    def feats(Rk, s, phi):
        z = np.log(np.clip(Rk, 1e-4, 1 - 1e-4) / (1 - np.clip(Rk, 1e-4, 1 - 1e-4)))
        return np.column_stack([z, s, phi, s * z, s * s])

    def predict(self, Rc, s, phi):
        P = np.stack([self.m[k].predict_proba(self.feats(Rc[:, k], s, phi))[:, 1] for k in range(K)], 1)
        return np.maximum.accumulate(np.clip(P, 1e-5, 1 - 1e-5), 1)


def fit_offset(prior, s, phi, y, idx, w=None):
    models = []
    for k in range(K):
        lr = LogisticRegression(C=10.0, max_iter=5000)
        lr.fit(WeightedOffset.feats(prior[idx, k], s[idx], phi[idx]), (y[idx] <= k).astype(int),
               sample_weight=None if w is None else w[idx])
        models.append(lr)
    return WeightedOffset(models)


def beta_of_rows(rows, iso, enc, dense=False):
    ranks = []
    for r in rows:
        m = r["in_crawl"]
        X = r["X"][m]
        R_ = apply_iso(iso, raw_cum(X, enc, dense))
        yy = r["y"][m]
        par = r["par"][m]
        for page, d in zip(r["fetched"], r["fetched_dist"]):
            if d not in (1, 2):
                continue
            sel = np.where(par == page)[0]
            if len(sel) == 0 or not (yy[sel] == d - 1).any():
                continue
            order = sel[np.argsort(-R_[sel, K - 1])]
            ranks.append(int(np.where(yy[order] == d - 1)[0][0]) + 1)
    return (float(np.mean(ranks)) if ranks else 1.0), len(ranks)


def edges_from(gp, nb):
    e = np.unique(np.quantile(np.asarray(gp, float), np.linspace(0, 1, nb + 1)))
    e[0] = 0.0
    e[-1] = np.inf
    return e


# ----------------------------------------------------------------------------- rule evaluation
def build_table(prior_, post_, y_, w_, qid_, beta_, edges, lam, rho, conf=0.95,
                smooth=None, weighted=False, min_count=5, region_mode="all", nboot=NBOOT_TABLE):
    gp = chain_index(prior_, beta_)[0]
    gq, Rq, Cq, _, _ = chain_index(post_, beta_)
    pay_q, act_q = realised_payoff(post_, y_, lam, beta_)
    pay_p, _ = realised_payoff(prior_, y_, lam, beta_)
    g = pay_q - pay_p
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, gp, side="right") - 1, 0, nb - 1)
    gain = np.zeros(nb)
    iidx = np.zeros(nb)
    pfetch = np.zeros(nb)
    cnt = np.zeros(nb, int)
    wsum = np.zeros(nb)
    for j in range(nb):
        m = b == j
        cnt[j] = int(m.sum())
        if cnt[j] < min_count:
            continue
        wj = w_[m] if weighted else np.ones(m.sum())
        wsum[j] = float(wj.sum())
        gain[j] = float((g[m] * wj).sum() / wj.sum())
        pfetch[j] = float((act_q[m] * wj).sum() / wj.sum())
        o = np.argsort(-gq[m])
        r = np.cumsum(Rq[m][o])
        cc = m.sum() * rho + np.cumsum(Cq[m][o])
        iidx[j] = float((r / cc).max())
    if smooth == "ma3":
        sm = gain.copy()
        for j in range(nb):
            lo, hi = max(0, j - 1), min(nb, j + 2)
            sm[j] = np.average(gain[lo:hi], weights=np.maximum(wsum[lo:hi], 1))
        gain = sm
    elif smooth == "iso":
        ir = IsotonicRegression(increasing=True, y_min=-1, y_max=1, out_of_bounds="clip")
        ir.fit(np.arange(nb), gain, sample_weight=np.maximum(wsum, 1))
        gain = ir.predict(np.arange(nb))
    uq_, inv = np.unique(qid_, return_inverse=True)
    S = np.zeros((len(uq_), nb))
    N = np.zeros((len(uq_), nb))
    wt = w_ if weighted else np.ones(len(g))
    np.add.at(S, (inv, b), g * wt)
    np.add.at(N, (inv, b), wt)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(uq_), (nboot, len(uq_)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.where(N[idx].sum(1) > 0, S[idx].sum(1) / N[idx].sum(1), 0.0)
    lcb = np.percentile(boot, 100 * (1 - conf), axis=0)
    region = (lcb > lam * rho) & (cnt >= min_count)
    if region_mode == "contiguous":
        pk = int(np.argmax(gain))
        reg = np.zeros(nb, bool)
        if region[pk]:
            lo = hi = pk
            while lo > 0 and region[lo - 1]:
                lo -= 1
            while hi < nb - 1 and region[hi + 1]:
                hi += 1
            reg[lo:hi + 1] = True
        region = reg
    return dict(edges=edges, gain=gain, lcb=lcb, region=region, insp_index=iidx,
                pfetch=pfetch, count=cnt, wsum=wsum, nb=nb)


def eval_rule(prior_, post_, y_, qid_, beta_, T, lam, rho, eval_mask, return_per_q=False):
    gp = chain_index(prior_, beta_)[0]
    b = np.clip(np.searchsorted(T["edges"], gp, side="right") - 1, 0, T["nb"] - 1)
    reg = T["region"][b] & eval_mask
    Ruse = np.where(reg[:, None], post_, prior_)
    pay_use, _ = realised_payoff(Ruse, y_, lam, beta_)
    pay_prior, _ = realised_payoff(prior_, y_, lam, beta_)
    d = (pay_use - pay_prior)
    net = d - lam * rho * reg.astype(float)
    out = dict(n=int(eval_mask.sum()), calls=float(reg[eval_mask].mean()),
               gross=float(d[eval_mask].mean()), net=float(net[eval_mask].mean()),
               net_per_call=float(net[eval_mask].mean() / max(reg[eval_mask].mean(), 1e-9)),
               net_per_100=float(100 * net[eval_mask].mean()),
               region_bins=int(T["region"].sum()), nb=int(T["nb"]),
               region_share=float(T["count"][T["region"]].sum() / max(T["count"].sum(), 1)))
    if return_per_q:
        uq, inv = np.unique(qid_[eval_mask], return_inverse=True)
        s = np.bincount(inv, net[eval_mask], len(uq))
        c = np.bincount(inv, minlength=len(uq)).astype(float)
        out["per_q"] = {str(q): float(v / n) for q, v, n in zip(uq, s, c)}
    return out


def crossfit(prior_, post_, y_, w_, qid_, beta_, lam, rho, fo, mask, conf=0.95, nb=20,
             smooth=None, weighted=False, region_mode="all"):
    folds = []
    for f in range(5):
        tr = mask & (fo != f)
        te = mask & (fo == f)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        edges = edges_from(chain_index(prior_[tr], beta_)[0], nb)
        T = build_table(prior_[tr], post_[tr], y_[tr], w_[tr], qid_[tr], beta_, edges,
                        lam, rho, conf, smooth, weighted, region_mode=region_mode)
        folds.append(eval_rule(prior_, post_, y_, qid_, beta_, T, lam, rho, te))
    keys = ("calls", "gross", "net", "net_per_call", "net_per_100", "region_bins", "region_share")
    return dict(folds=folds, mean={k: float(np.mean([o[k] for o in folds])) for k in keys},
                n=int(sum(o["n"] for o in folds)))


def paired_ci(a, b, nboot=NBOOT_FINAL, seed=0):
    """Paired cluster bootstrap over questions for mean(a - b)."""
    q = sorted(set(a) & set(b))
    if not q:
        return None
    d = np.array([a[x] - b[x] for x in q], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (nboot, len(d)))
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
    return dict(mean=float(d.mean()), lo95=float(lo), hi95=float(hi), p=float(p), nq=int(len(q)))


def split_eval(prior_, post_, y_, w_, qid_, beta_, lam, rho, tr_mask, te_mask, conf=0.95, nb=20,
               smooth=None, weighted=False, region_mode="all", return_per_q=False):
    edges = edges_from(chain_index(prior_[tr_mask], beta_)[0], nb)
    T = build_table(prior_[tr_mask], post_[tr_mask], y_[tr_mask], w_[tr_mask], qid_[tr_mask],
                    beta_, edges, lam, rho, conf, smooth, weighted, region_mode=region_mode)
    ev = eval_rule(prior_, post_, y_, qid_, beta_, T, lam, rho, te_mask, return_per_q=return_per_q)
    ev["table"] = dict(region=[bool(x) for x in T["region"]], gain=[float(x) for x in T["gain"]],
                       lcb=[float(x) for x in T["lcb"]], count=[int(x) for x in T["count"]],
                       edges=[float(x) for x in T["edges"]])
    return ev


# ----------------------------------------------------------------------------- contexts
def load_ctx(regime):
    if regime == "trained":
        md = os.path.join(D, "models")
        kind = "index_C"
        modeldir = md
    else:
        md = os.path.join(D, "models_weak")
        kind = "index_weakC"
        modeldir = md
    post_path = os.path.join(modeldir, "posterior.pkl")
    p = pickle.load(open(post_path, "rb"))
    cal = p["cal"]
    levels = {g["qid"]: g["level"] for g in load_goals("cal")}
    qid = np.array(cal["qid"])
    hard = np.array([levels[q] == "hard" for q in qid])
    y = np.array(cal["y"])
    s = np.array(cal["s"])
    phi = np.array(cal["phi"])
    w = np.array(cal["w"])
    prior_full = np.array(cal["prior"])
    post_full = np.array(cal["post_oof"])
    prior_infold = np.array(cal["prior_infold"])

    rows = []
    for f in sorted(glob.glob(os.path.join(D, "rows", f"cal_{kind}_s*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    calp = pickle.load(open(os.path.join(modeldir, "calib.pkl"), "rb"))
    beta = calp["beta"]
    dense = bool(calp.get("dense"))
    enc = None if dense else lgb.Booster(model_file=os.path.join(modeldir, "encoder.txt"))
    lam = json.load(open(os.path.join(modeldir, "lambda.json")))["25"]["lam"]

    # replicate the posterior sample exactly (same rng and quota as fit_posterior.py)
    per_crawl = 30 if regime == "trained" else 8
    rng = np.random.default_rng(1)
    pool = []
    for r in rows:
        m = r["in_crawl"]
        pr = r["prior"][m]
        gg = chain_index(pr, beta)[0]
        for j in range(len(gg)):
            pool.append((r["gid"], int(r["par"][m][j]), int(r["tgt"][m][j]), pr[j],
                         int(r["y"][m][j]), gg[j], r["X"][m][j]))
    gall = np.array([x[5] for x in pool])
    e0 = np.unique(np.quantile(gall, np.linspace(0, 1, 21)))
    e0[0], e0[-1] = 0.0, np.inf
    pbin = np.clip(np.searchsorted(e0, gall, side="right") - 1, 0, len(e0) - 2)
    by = {}
    for i, x in enumerate(pool):
        by.setdefault(x[0], []).append(i)
    sample = []
    for gid, idxs in by.items():
        idxs = np.array(idxs)
        bins = pbin[idxs]
        chosen = []
        present = rng.permutation(np.unique(bins))
        q = per_crawl
        while q > 0 and len(chosen) < len(idxs):
            for b in present:
                cand = [i for i in idxs[bins == b] if i not in chosen]
                if cand and q > 0:
                    chosen.append(int(rng.choice(cand)))
                    q -= 1
        sample += chosen
    sample = np.array(sample)
    assert len(sample) == len(qid), (len(sample), len(qid))
    assert np.allclose(np.stack([pool[i][3] for i in sample]), prior_infold)
    assert np.all(np.array([pool[i][0].rsplit("_h", 1)[0] for i in sample]) == qid)
    X = np.stack([pool[i][6] for i in sample])
    parent = np.array([pool[i][1] for i in sample])
    target = np.array([pool[i][2] for i in sample])

    # the original fold assignment: critically, post_oof is out-of-fold with respect to this
    uq = np.unique(qid)
    fold = {q: i % 5 for i, q in enumerate(rng.permutation(uq))}
    fo = np.array([fold[q] for q in qid])

    # full-population prior index for fixed bin edges
    gp_full_all = np.concatenate([chain_index(r["prior"][r["in_crawl"]], beta)[0] for r in rows])

    return dict(regime=regime, p=p, cal=cal, levels=levels, qid=qid, hard=hard, y=y, s=s,
                phi=phi, w=w, prior_full=prior_full, post_full=post_full,
                prior_infold=prior_infold, rows=rows, beta=beta, enc=enc, dense=dense,
                lam=float(lam), X=X, parent=parent, target=target, fo=fo,
                gp_full_all=gp_full_all, sample=sample, per_crawl=per_crawl)


def make_full(ctx):
    return dict(name="full", prior=ctx["prior_full"], post=ctx["post_full"], beta=ctx["beta"],
                lam=ctx["lam"], mask=np.ones(len(ctx["qid"]), bool), gp_all=ctx["gp_full_all"],
                note="deployed calibration")


def make_hard(ctx):
    rows = ctx["rows"]
    levels = ctx["levels"]
    hard_rows = [r for r in rows if levels[r["gid"].rsplit("_h", 1)[0]] == "hard"]
    Xh = np.vstack([r["X"][r["in_crawl"]] for r in hard_rows])
    yh = np.concatenate([r["y"][r["in_crawl"]] for r in hard_rows])
    cum_h = raw_cum(Xh, ctx["enc"], ctx["dense"])
    iso_h = fit_iso(cum_h, yh)
    prior_h_all = apply_iso(iso_h, cum_h)
    cum_s = raw_cum(ctx["X"], ctx["enc"], ctx["dense"])
    prior_h = apply_iso(iso_h, cum_s)
    beta_h, npg = beta_of_rows(hard_rows, iso_h, ctx["enc"], ctx["dense"])
    hard = ctx["hard"]
    post_h = np.zeros_like(prior_h)
    for f in range(5):
        tr = hard & (ctx["fo"] != f)
        te = hard & (ctx["fo"] == f)
        if te.sum() == 0:
            continue
        post_h[te] = fit_offset(prior_h, ctx["s"], ctx["phi"], ctx["y"], tr).predict(
            prior_h[te], ctx["s"][te], ctx["phi"][te])
    gp_h_all = chain_index(prior_h_all, beta_h)[0]
    # hard lambda from hard validation questions, fallback to full lambda
    lam = ctx["lam"]
    try:
        run = load("val", "index_r1_weak_rho0.1" if ctx["regime"] == "weak" else "index_r1_rho0.1")
        hardq = {q for q, l in levels.items() if l == "hard"}
        last = []
        for r in run.values():
            if r.get("qid") not in hardq:
                continue
            f = [e for e in r["events"] if e[0] == "F" and e[1] <= 25 + 1e-9]
            if f:
                last.append(f[-1][5])
        if len(last) >= 10:
            lam = float(np.median(last))
    except Exception:
        pass
    return dict(name="hard", prior=prior_h, post=post_h, beta=beta_h, lam=lam, mask=hard,
                gp_all=gp_h_all, note=f"hard-only calibration, beta={beta_h:.4f}, n_pages={npg}")


def make_weighted(ctx, hard_weight):
    rows = ctx["rows"]
    level_w_rows = np.concatenate([
        np.full(int(r["in_crawl"].sum()),
                1.0 if ctx["levels"][r["gid"].rsplit("_h", 1)[0]] == "hard" else hard_weight)
        for r in rows])
    X_all = np.vstack([r["X"][r["in_crawl"]] for r in rows])
    y_all = np.concatenate([r["y"][r["in_crawl"]] for r in rows])
    cum_all = raw_cum(X_all, ctx["enc"], ctx["dense"])
    iso_w = fit_iso(cum_all, y_all, w=level_w_rows)
    prior_w_all = apply_iso(iso_w, cum_all)
    cum_s = raw_cum(ctx["X"], ctx["enc"], ctx["dense"])
    prior_w = apply_iso(iso_w, cum_s)
    beta_w, _ = beta_of_rows(rows, iso_w, ctx["enc"], ctx["dense"])
    level_w = np.where(ctx["hard"], 1.0, hard_weight)
    w_comb = ctx["w"] * level_w
    post_w = np.zeros_like(prior_w)
    for f in range(5):
        tr = ctx["fo"] != f
        te = ctx["fo"] == f
        post_w[te] = fit_offset(prior_w, ctx["s"], ctx["phi"], ctx["y"], tr, w_comb).predict(
            prior_w[te], ctx["s"][te], ctx["phi"][te])
    gp_w_all = chain_index(prior_w_all, beta_w)[0]
    return dict(name=f"weighted{hard_weight:g}", prior=prior_w, post=post_w, beta=beta_w,
                lam=ctx["lam"], mask=np.ones(len(ctx["qid"]), bool), gp_all=gp_w_all,
                note=f"difficulty-weighted calibration, hard weight 1.0, other {hard_weight:g}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="both", choices=("trained", "weak", "both"))
    ap.add_argument("--out", default=os.path.join(R, "tables", "twocost_ablation.json"))
    a = ap.parse_args()
    regimes = ("trained", "weak") if a.regime == "both" else (a.regime,)
    out = {"protocol": {
        "split_seed": SEED_SPLIT,
        "tuning": "50% of calibration questions; tables built only on this half",
        "heldout": "other 50% of calibration questions; selected rule applied without refitting",
        "selection": "one-standard-error rule on the tuning half: among variants within one SE of the best tuning net, take the fewest calls; the deployed rule wins ties and is frozen unless the best tuning net beats it by its own SE",
        "h1": "hard-calibrated table beats deployed table on hard held-out links",
        "h2": "lower confidence or shrinkage beats deployed rule on held-out links",
        "h3": "different bin count or smoothing beats deployed rule on held-out links",
        "no_new_judge_calls": True,
        "caveat": "offline calibration-link evaluation; crawl-level confirmation requires the judge backend",
    }, "regimes": {}}

    for regime in regimes:
        print("=== loading", regime, flush=True)
        ctx = load_ctx(regime)
        qids = np.unique(ctx["qid"])
        rng = np.random.default_rng(SEED_SPLIT)
        perm = rng.permutation(qids)
        tune_q = set(perm[: len(perm) // 2])
        tr_mask = np.array([q in tune_q for q in ctx["qid"]])
        te_mask = ~tr_mask
        print(f"questions tune={len(tune_q)} heldout={len(qids)-len(tune_q)} links tune={tr_mask.sum()} heldout={te_mask.sum()} hard={ctx['hard'].sum()}", flush=True)

        variants = [make_full(ctx), make_hard(ctx),
                    make_weighted(ctx, 0.25), make_weighted(ctx, 0.5)]
        # H2 / H3 grids on the full calibration and on hard calibration
        base_variants = [variants[0], variants[1]]
        for v in base_variants:
            for c in CONF_LEVELS:
                for nb in BIN_LEVELS:
                    for sm in (None, "ma3", "iso"):
                        if v["name"] == "hard" and c != 0.95 and nb != 20:
                            continue
                        if v["name"] == "full" and sm is not None and (c != 0.95 or nb != 20):
                            continue
                        variants.append(dict(v, name=f"{v['name']}_c{int(c*100)}_b{nb}" + (f"_{sm}" if sm else ""),
                                             conf=c, nb=nb, smooth=sm))
        for v in variants[2:4]:
            for c in (0.8, 0.95):
                variants.append(dict(v, name=f"{v['name']}_c{int(c*100)}_b20", conf=c, nb=20, smooth=None))

        # evaluate every variant on the tuning half and the held-out half.
        # The test split is 100% hard, so the primary population is hard links; every variant is
        # compared on the same hard population, while its table is built on its own calibration
        # population (all levels for H2/H3, hard only for H1, weighted for the weighted variant).
        for rho in RHOS:
            rec = {"rho": rho, "variants": {}, "base": None, "selected": None, "h1": None}
            for v in variants:
                name = v["name"]
                conf = v.get("conf", 0.95)
                nb = v.get("nb", 20)
                sm = v.get("smooth", None)
                train_mask_v = tr_mask & v["mask"]
                try:
                    tune = split_eval(v["prior"], v["post"], ctx["y"], ctx["w"], ctx["qid"],
                                      v["beta"], v["lam"], rho, train_mask_v,
                                      tr_mask & ctx["hard"], conf, nb, sm,
                                      weighted=("weighted" in name), region_mode="all",
                                      return_per_q=True)
                    held = split_eval(v["prior"], v["post"], ctx["y"], ctx["w"], ctx["qid"],
                                      v["beta"], v["lam"], rho, train_mask_v,
                                      te_mask & ctx["hard"], conf, nb, sm,
                                      weighted=("weighted" in name), region_mode="all",
                                      return_per_q=True)
                    held_all = None
                    if bool(v["mask"].all()):
                        held_all = split_eval(v["prior"], v["post"], ctx["y"], ctx["w"], ctx["qid"],
                                              v["beta"], v["lam"], rho, train_mask_v, te_mask,
                                              conf, nb, sm, weighted=("weighted" in name),
                                              region_mode="all", return_per_q=True)
                except Exception as e:
                    rec["variants"][name] = {"error": repr(e)}
                    continue
                tq = np.array(list(tune["per_q"].values()))
                se = float(tq.std(ddof=1) / np.sqrt(max(len(tq), 1))) if len(tq) > 1 else 0.0
                rec["variants"][name] = {
                    "tune_hard": {k: tune[k] for k in ("n", "calls", "net", "net_per_call", "region_bins", "nb", "region_share")},
                    "tune_se": se,
                    "heldout_hard": {k: held[k] for k in ("n", "calls", "net", "net_per_call", "net_per_100", "region_bins", "nb", "region_share")},
                    "heldout_all": None if held_all is None else {k: held_all[k] for k in ("n", "calls", "net", "net_per_call", "net_per_100", "region_bins", "nb", "region_share")},
                    "tune_per_q": tune["per_q"],
                    "heldout_per_q": held["per_q"],
                    "lam": v["lam"], "note": v.get("note", "")}
                if name == "full_c95_b20":
                    rec["base"] = rec["variants"][name]
            # H1 is the hard table against the full table on the same held-out hard links
            vh = variants[1]
            try:
                train_h = tr_mask & vh["mask"]
                h1_held = split_eval(vh["prior"], vh["post"], ctx["y"], ctx["w"], ctx["qid"],
                                     vh["beta"], vh["lam"], rho, train_h,
                                     te_mask & ctx["hard"], 0.95, 20, None, region_mode="all")
                vf = variants[0]
                base_hard = split_eval(vf["prior"], vf["post"], ctx["y"], ctx["w"], ctx["qid"],
                                       vf["beta"], vf["lam"], rho, tr_mask,
                                       te_mask & ctx["hard"], 0.95, 20, None, region_mode="all")
                rec["h1"] = {"hard_table": h1_held, "base_table_on_hard": base_hard,
                             "delta": h1_held["net"] - base_hard["net"]}
            except Exception as e:
                rec["h1"] = {"error": repr(e)}
            # One-standard-error selection on the tuning half (hard links): among variants within
            # one SE of the best tuning net, take the one with the fewest calls; the deployed rule
            # wins ties.  If the best tuning net is not better than the deployed rule's by its own
            # SE, freeze the deployed rule (a null).
            cand = []
            for name, r in rec["variants"].items():
                if "tune_hard" not in r:
                    continue
                t = r["tune_hard"]
                if t["region_bins"] < 1 or t["region_bins"] > t["nb"] - 1 or t["calls"] > 0.5:
                    continue
                cand.append((t["net"], r["tune_se"], t["calls"], name))
            if cand:
                best_net, best_se, _, best = max(cand, key=lambda x: x[0])
                base_net = rec["base"]["tune_hard"]["net"] if rec["base"] else -1e9
                if best_net - base_net > best_se:
                    ok = [c for c in cand if c[0] >= best_net - best_se]
                    ok.sort(key=lambda x: (x[2], 0 if x[3] == "full_c95_b20" else 1, -x[0]))
                    rec["selected"] = ok[0][3]
                else:
                    rec["selected"] = "full_c95_b20"
            else:
                rec["selected"] = "full_c95_b20"
            sel = rec["variants"][rec["selected"]]["heldout_hard"]
            base = rec["variants"]["full_c95_b20"]["heldout_hard"]
            sel_q = rec["variants"][rec["selected"]]["heldout_per_q"]
            base_q = rec["variants"]["full_c95_b20"]["heldout_per_q"]
            rec["selected_vs_base"] = {
                "selected": rec["selected"],
                "heldout_net_selected": sel["net"],
                "heldout_net_base": base["net"],
                "heldout_delta": sel["net"] - base["net"],
                "heldout_calls_selected": sel["calls"],
                "heldout_calls_base": base["calls"],
                "paired_ci": paired_ci(sel_q, base_q),
            }
            # Pre-declared mechanism contrasts on held-out hard links.
            contrasts = {
                "H1_hard95_vs_base": "hard_c95_b20",
                "H2_conf50_vs_base": "full_c50_b20",
                "H2_conf80_vs_base": "full_c80_b20",
                "H2_conf90_vs_base": "full_c90_b20",
                "H3_bins10_vs_base": "full_c95_b10",
                "H3_bins40_vs_base": "full_c95_b40",
                "H2H3_conf50_bins40_vs_base": "full_c50_b40",
                "H3_smooth_ma3_vs_base": "full_c95_b20_ma3",
                "H3_smooth_iso_vs_base": "full_c95_b20_iso",
            }
            rec["contrasts"] = {}
            for label, cname in contrasts.items():
                if cname in rec["variants"] and "heldout_per_q" in rec["variants"][cname]:
                    rec["contrasts"][label] = paired_ci(rec["variants"][cname]["heldout_per_q"], base_q)
            out["regimes"].setdefault(regime, {})[str(rho)] = rec
            print(f"{regime} rho={rho} selected={rec['selected']} "
                  f"heldout-hard net sel={sel['net']:+.6f} base={base['net']:+.6f} "
                  f"delta={sel['net']-base['net']:+.6f} calls sel={sel['calls']:.4f} base={base['calls']:.4f} "
                  f"CI=[{rec['selected_vs_base']['paired_ci']['lo95']:+.6f},{rec['selected_vs_base']['paired_ci']['hi95']:+.6f}]",
                  flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
