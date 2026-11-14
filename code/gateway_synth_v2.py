"""Extended controlled gateway test (v2) for Proposition 2.

Adds a fine dependence grid, wider out-degree and a second depth to the original
gateway_synth.py, and places the Wikipedia graph on the same axis through its measured
seed agreement (0.948).  Pure simulation; no judge call.

The dependence parameter `delta` is how much the one-hop score already favours the gateway
link when the gold page is not yet adjacent.  delta=0 means the one-hop score cannot separate
the gateway from a decoy; delta large means it can.  The index should win at delta=0 and
converge to myopic as delta grows.  Seed agreement is the observable counterpart: high
agreement means the one-hop score already picks the gateway, so the index has nothing to add.

Writes results/theory/gateway_synth_v2.json and manuscript/figures/fig_gateway.pdf.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gateway_synth import Goal, crawl, seed_agreement, boot  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "theory", "gateway_synth_v2.json")
FIG = os.path.join(os.path.dirname(HERE), "..", "manuscript", "figures", "fig_gateway.pdf")

DELTAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
WIDTHS = (3, 5, 7)
DEPTHS = (2, 3)
SIGMAS = (0.0, 0.15)
N = 400
WIKI_AGREEMENT = 0.948


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
    cells = []
    total = len(DEPTHS) * len(WIDTHS) * len(DELTAS) * len(SIGMAS)
    k = 0
    for h, width, delta, sigma in itertools.product(DEPTHS, WIDTHS, DELTAS, SIGMAS):
        c = cell(h, width, delta, sigma)
        cells.append(c)
        k += 1
        d = c["diff"]
        print(f"[{k}/{total}] h={h} m={width} delta={delta:.2f} sigma={sigma:.2f}: "
              f"index {c['index_mean']:.3f} myopic {c['myopic_mean']:.3f} "
              f"diff {d['mean']:+.3f} [{d['lo95']:+.3f},{d['hi95']:+.3f}] "
              f"agree {c['agreement']:.2f}" + ("  TOST" if d["equiv"] else ""), flush=True)

    # Wikipedia point: interpolate delta at which seed agreement equals the measured 0.948,
    # on the primary curve (h=3, width=5, sigma=0).
    prim = sorted([c for c in cells if c["h"] == 3 and c["width"] == 5 and c["sigma"] == 0.0],
                  key=lambda c: c["delta"])
    ag = np.array([c["agreement"] for c in prim])
    dl = np.array([c["delta"] for c in prim])
    diff = np.array([c["diff"]["mean"] for c in prim])
    wiki = dict(agreement=WIKI_AGREEMENT, delta_hat=None, diff_hat=None, diff_lo=None, diff_hi=None)
    if ag.min() <= WIKI_AGREEMENT <= ag.max():
        # agreement is decreasing in delta; invert by linear interpolation on the monotone segment
        o = np.argsort(ag)
        wiki["delta_hat"] = float(np.interp(WIKI_AGREEMENT, ag[o], dl[o]))
        wiki["diff_hat"] = float(np.interp(WIKI_AGREEMENT, ag[o], diff[o]))
        # conservative interval: interpolate the lower/upper bounds at the same agreement
        lo = np.array([c["diff"]["lo95"] for c in prim])
        hi = np.array([c["diff"]["hi95"] for c in prim])
        wiki["diff_lo"] = float(np.interp(WIKI_AGREEMENT, ag[o], lo[o]))
        wiki["diff_hi"] = float(np.interp(WIKI_AGREEMENT, ag[o], hi[o]))

    rec = dict(config=dict(beta=2.5846579129232894, budget=25, K=3, n_per_cell=N,
                           depths=list(DEPTHS), widths=list(WIDTHS), deltas=list(DELTAS),
                           sigmas=list(SIGMAS), wiki_agreement=WIKI_AGREEMENT,
                           note="synthetic mechanism check; fine dependence grid; no judge"),
               cells=cells, wikipedia=wiki)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w"), indent=1)
    print("wiki", wiki)
    print("wrote", OUT)
    make_fig(rec)


def make_fig(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 1.40), constrained_layout=True)
    colors = {2: "#1f77b4", 3: "#d62728"}
    styles = {3: "--", 5: "-", 7: ":"}
    for hh in DEPTHS:
        for ww in WIDTHS:
            cs = sorted([c for c in rec["cells"] if c["h"] == hh and c["width"] == ww and c["sigma"] == 0.0],
                        key=lambda c: c["delta"])
            x = np.array([c["delta"] for c in cs])
            y = np.array([c["diff"]["mean"] for c in cs])
            lo = np.array([c["diff"]["lo95"] for c in cs])
            hi = np.array([c["diff"]["hi95"] for c in cs])
            ax.plot(x, y, styles[ww], color=colors[hh], lw=1.1, label=f"h={hh}, width={ww}")
            ax.fill_between(x, lo, hi, color=colors[hh], alpha=0.10, linewidth=0)
    wiki = rec["wikipedia"]
    if wiki["delta_hat"] is not None:
        ax.axvline(wiki["delta_hat"], color="k", ls=":", lw=0.9)
        ax.annotate("Wikipedia\n(agreement 0.948)", xy=(wiki["delta_hat"], 0.30),
                    xytext=(wiki["delta_hat"] + 0.015, 0.30), fontsize=6.2, va="center")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("one-hop informativeness $\\delta$", fontsize=7.5)
    ax.set_ylabel("recall(index) $-$ recall(myopic)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.6, ncol=2, frameon=False, loc="upper right",
              handlelength=1.4, columnspacing=0.8, borderaxespad=0.2)
    ax.set_ylim(-0.03, 0.80)
    fig.savefig(FIG, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fig-only":
        rec = json.load(open(OUT))
        make_fig(rec)
    else:
        main()
