"""Controlled gateway test for Proposition 2 (exploratory).

Question: when the one-hop score cannot separate a gateway link (whose subtree reaches a gold page)
from decoys, but the multi-hop layer probabilities can, does the subtree index recover the ordering?
And does the advantage disappear when the one-hop score already identifies the gateway?

Design. A synthetic layered tree (depth h = 2, 3, 4), out-degree 1 + m, m in {2, 5}. At every page on
the gold path, one link continues the path and m links lead to dead subtrees with no gold. With
m = 2 and budget B = 25 the expected cost of a random traversal, h(1 + m) <= 12, is well inside B, so
the myopic rule cannot fail and the cell has no power; m = 5 is the primary width because h(1 + m)
reaches B at h = 4 and leaves only a thin margin at h = 3. Each link carries a true cumulative vector
R = (R1, R2, R3) = P(nearest gold within k fetches through the link). The observed one-hop score is R1
plus N(0, 0.05); on a path link whose gold is not yet adjacent, an extra `delta` is added to R1
(one-hop informativeness). The observed R2/R3 are the true values plus N(0, sigma), made monotone
(multi-hop informativeness); all three entries carry the 0.05 measurement noise. Both policies see
the same observed vectors: the myopic rule picks the largest R1; the index picks the largest
chain_index(R, beta), with beta the calibrated per-level cost from the real model
(2.5846579129232894). Each crawl fetches until budget 25; recall is the fraction of {P1, P2} found.
Goals are paired across policies; intervals are a percentile bootstrap over goals (10,000 resamples,
seed 0), TOST margin 0.02.

Pre-declared criteria (stated before the first run):
  success      delta=0, sigma=0 at width 5, h in {2, 3}: index - myopic >= +0.02 with the 95%
               interval above 0, and top-link agreement at the seed below 50%;
  equivalence  delta=0.3, sigma=0: TOST at 0.02 over all widths and depths (the index must not add
               measurable recall when the one-hop score already identifies the gateway);
  boundary     h=4, delta=0, sigma=0: the first gateway is 4 fetches away, beyond the K=3 horizon, so
               the observed layers carry no signal at the seed and no advantage is expected;
  failure      TOST equivalence at delta=0, sigma=0, width 5, h in {2, 3}.
This is a mechanism check with exact/perturbed probabilities on a synthetic graph. It is not a
confirmatory test and says nothing about the Wikipedia graph.

Writes results/theory/gateway_synth.json; reads nothing.
"""
import itertools
import json
import os
import zlib

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl import chain_index

BETA = 2.5846579129232894
BUDGET = 25
NB = 10000
SEED = 0
DELTA = 0.02
N = 400
WIDTHS = (2, 5)
DEPTHS = (2, 3, 4)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "theory", "gateway_synth.json")


class Link:
    __slots__ = ("child", "r", "idx")

    def __init__(self, child, r):
        self.child = child
        self.r = r
        self.idx = float(chain_index(r[None, :], BETA)[0][0])


class Goal:
    """Lazily generated layered tree; children are cached so both policies see identical links."""

    def __init__(self, gid, h, width, delta, sigma):
        self.gid = gid
        self.h = h
        self.width = width
        self.delta = delta
        self.sigma = sigma
        self.cache = {}

    def _kind(self, node):
        if node == "seed":
            return 0, "seed"
        if node == "p2":
            return self.h + 1, "p2"
        if node.startswith("g"):
            return int(node[1:]), "path"
        return int(node[1:].split("-")[0]), "dead"

    def children(self, node):
        if node in self.cache:
            return self.cache[node]
        d, kind = self._kind(node)
        if kind == "seed" or (kind == "path" and d < self.h):
            kids = [f"g{d + 1}"] + [f"x{d + 1}-{i}" for i in range(self.width)]
        elif kind == "path" and d == self.h:          # P1: one link to P2, two decoys
            kids = ["p2"] + [f"x{d + 1}-{i}" for i in (0, 1)]
        elif kind == "dead" and d < self.h + 2:
            kids = [f"x{d + 1}-{i}" for i in (0, 1, 2)]
        else:
            kids = []
        rng = np.random.default_rng([zlib.crc32(node.encode()), self.gid, 7])
        order = rng.permutation(len(kids))
        out = []
        for j in order:
            child = kids[j]
            cd, _ = self._kind(child)
            path_link = kind in ("seed", "path") and (child == "p2" or child.startswith("g"))
            true = np.zeros(3)
            if path_link:
                dist = 1 if child == "p2" else self.h - d          # fetches from here to nearest gold
                true = (np.arange(1, 4) >= dist).astype(float)
            r = true + rng.normal(0.0, 0.05, 3)
            if path_link and true[0] == 0 and self.delta:
                r[0] += self.delta
            if self.sigma:
                r[1:] += rng.normal(0.0, self.sigma, 2)
            r = np.maximum.accumulate(np.clip(r, 1e-6, 1 - 1e-6))
            out.append(Link(child, r))
        self.cache[node] = out
        return out


def pick(links, mode):
    key = [l.r[0] for l in links] if mode == "myopic" else [l.idx for l in links]
    return int(np.argmax(key))


def crawl(goal, mode):
    frontier = list(goal.children("seed"))
    seen = {"seed"}
    found = set()
    cost = 0
    while cost < BUDGET and frontier:
        i = pick(frontier, mode)
        link = frontier.pop(i)
        cost += 1
        child = link.child
        if child == "p2":
            found.add("P2")
            continue
        if child.startswith("g") and goal._kind(child)[0] == goal.h:
            found.add("P1")                            # this is P1; it reveals the P2 link
        if child not in seen:
            seen.add(child)
            frontier.extend(goal.children(child))
    return len(found) / 2.0


def seed_agreement(goal):
    links = goal.children("seed")
    return float(pick(links, "myopic") == pick(links, "index"))


def boot(d):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (NB, len(d)))
    m = d[idx].mean(1)
    lo95, hi95 = np.percentile(m, [2.5, 97.5])
    lo90, hi90 = np.percentile(m, [5, 95])
    p = float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
    return dict(mean=float(np.mean(d)), lo95=float(lo95), hi95=float(hi95),
                lo90=float(lo90), hi90=float(hi90), p=p,
                equiv=bool(lo90 > -DELTA and hi90 < DELTA), n=int(len(d)))


def cell(h, width, delta, sigma, n=N):
    r_i, r_m, agree = [], [], []
    for gid in range(n):
        g = Goal(gid, h, width, delta, sigma)
        r_m.append(crawl(g, "myopic"))
        r_i.append(crawl(g, "index"))
        agree.append(seed_agreement(g))
    ri, rm = np.array(r_i), np.array(r_m)
    b = boot(ri - rm)
    return dict(h=h, width=width, delta=delta, sigma=sigma, n=n,
                index_mean=float(ri.mean()), myopic_mean=float(rm.mean()),
                diff=b, agreement=float(np.mean(agree)))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"refusing to overwrite {OUT}")
    cells = []
    for h, width, delta, sigma in itertools.product(DEPTHS, WIDTHS, (0.0, 0.15, 0.3), (0.0, 0.15, 0.3)):
        c = cell(h, width, delta, sigma)
        cells.append(c)
        d = c["diff"]
        print(f"h={h} m={width} delta={delta:.2f} sigma={sigma:.2f}: "
              f"index {c['index_mean']:.3f} myopic {c['myopic_mean']:.3f} "
              f"diff {d['mean']:+.3f} [{d['lo95']:+.3f},{d['hi95']:+.3f}] "
              f"p {d['p']:.3f} agree {c['agreement']:.2f}"
              + ("  TOST" if d["equiv"] else ""), flush=True)
    rec = dict(config=dict(beta=BETA, budget=BUDGET, K=3, n_per_cell=N, nboot=NB, seed=SEED,
                           delta_tost=DELTA, depths=list(DEPTHS), widths=list(WIDTHS),
                           noise_onehop=0.05,
                           note="synthetic mechanism check; exploratory; exact/perturbed "
                                "multi-hop probabilities; no judge"),
               cells=cells)
    json.dump(rec, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
