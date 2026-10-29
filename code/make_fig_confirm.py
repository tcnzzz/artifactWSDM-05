"""Figure 1: confirmatory recall gain over the scorer at the two prices, both regimes.

Reads results/tables/round1_confirm.json (saved output of round1_confirm.py) and writes
manuscript/figures/fig_confirm.pdf. Plots no computed quantity of its own beyond the saved
cell means and 95% intervals.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ART, "..", "manuscript")
conf = json.load(open(os.path.join(ART, "results", "tables", "round1_confirm.json")))
cells = conf["cells"]

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "legend.fontsize": 6.6, "pdf.fonttype": 42, "axes.titlesize": 8.5})
COL = {"ours": "#c0392b", "band": "#1b4f72", "topm": "#f39c12", "thompson": "#148f77"}
LAB = {"ours": "TwoCost", "band": "band cascade", "topm": "top-$m$", "thompson": "Thompson"}
MK = {"ours": "o", "band": "s", "topm": "^", "thompson": "D"}
DODGE = {"ours": 0.94, "band": 0.98, "topm": 1.02, "thompson": 1.06}

fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.0), sharex=True)
for ax, reg, title in zip(axes, ("trained", "zeroshot"),
                          ("trained scorer", "zero-shot scorer")):
    for p in ("ours", "band", "topm", "thompson"):
        xs, ys, lo, hi = [], [], [], []
        for rho in ("0.1", "1"):
            c = cells[reg][rho][p]["vs_scorer"]
            xs.append(float(rho) * DODGE[p])
            ys.append(c["mean"]); lo.append(c["lo95"]); hi.append(c["hi95"])
        xs = np.array(xs); ys = np.array(ys)
        ax.errorbar(xs, ys, yerr=[ys - np.array(lo), np.array(hi) - ys], fmt=MK[p],
                    color=COL[p], ms=3.2, lw=0.8, capsize=1.2, elinewidth=0.7,
                    alpha=0.9, label=LAB[p])
    ax.axhline(0.0, color="#555555", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.075, 1.35)
    ax.set_xticks([0.1, 1.0]); ax.set_xticklabels(["0.1", "1"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.4)
axes[0].legend(frameon=False, ncol=2, loc="lower left", handlelength=1.0, columnspacing=0.8)
axes[1].set_xlabel(r"price ratio $\rho = c_l/c_f$")
fig.supylabel(r"$\Delta$ recall vs scorer", fontsize=7, x=0.02)
fig.tight_layout(pad=0.25, rect=(0.055, 0, 1, 1))
out = os.path.join(MS, "figures", "fig_confirm.pdf")
fig.savefig(out, metadata={"Creator": None, "CreationDate": None})
plt.close(fig)
print("wrote", out)
