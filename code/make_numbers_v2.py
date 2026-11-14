"""Emit manuscript/numbers_v2.tex from the offline stress-test results.

Reads only saved result files; makes no judge call.  Run from the repository root:
    PYTHONPATH=artifact/code .venv/bin/python artifact/code/make_numbers_v2.py
"""
import json
import os

from common import R

MAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "manuscript")
OUT = os.path.join(MAN, "numbers_v2.tex")
EXP = os.path.join(R, "exploratory")
TAB = os.path.join(R, "tables")
TH = os.path.join(R, "theory")


def load(p):
    with open(p) as f:
        return json.load(f)


def f3(x):
    return f"{x:+.3f}"


def f4(x):
    return f"{x:+.4f}"


def f6(x):
    v = f"{x:+.6f}"
    return v


def ci(d, nd=3):
    return f"[{d['lo95']:+.{nd}f}, {d['hi95']:+.{nd}f}]"


def main():
    abl_t = load(os.path.join(EXP, "twocost_ablation_trained.json"))
    abl_w = load(os.path.join(EXP, "twocost_ablation_weak.json"))
    scr = load(os.path.join(EXP, "scorer_strength_screen.json"))["cells"]
    gw = load(os.path.join(TH, "gateway_synth_v2.json"))
    th = load(os.path.join(TAB, "thompson_diagnosis.json"))
    pc = load(os.path.join(EXP, "prop1_confirm_cached.json"))["cells"]
    se = load(os.path.join(EXP, "sensitivity.json"))

    M = {}

    # ---- TwoCost v2 ablation -------------------------------------------------
    tr = abl_t["regimes"]["trained"]
    wk = abl_w["regimes"]["weak"]
    sel_t = tr["0.1"]["selected_vs_base"]
    sel_w = wk["0.1"]["selected_vs_base"]
    M["AblTrainedDelta"] = f6(sel_t["heldout_delta"])
    M["AblTrainedCI"] = f"[{sel_t['paired_ci']['lo95']:+.5f}, {sel_t['paired_ci']['hi95']:+.5f}]"
    M["AblTrainedCalls"] = f"{sel_t['heldout_calls_base']:.3f}"
    M["AblWeakDelta"] = f6(sel_w["heldout_delta"])
    M["AblWeakCI"] = f"[{sel_w['paired_ci']['lo95']:+.5f}, {sel_w['paired_ci']['hi95']:+.5f}]"
    M["AblWeakCallsSel"] = f"{sel_w['heldout_calls_selected']:.3f}"
    M["AblWeakCallsBase"] = f"{sel_w['heldout_calls_base']:.3f}"
    M["AblWeakNetBase"] = f"{sel_w['heldout_net_base']:+.5f}"
    def contrast(reg, key):
        d = reg["0.1"]["contrasts"][key]
        return f"{d['mean']:+.5f} [{d['lo95']:+.5f}, {d['hi95']:+.5f}]"
    for tag, reg in (("Trained", tr), ("Weak", wk)):
        for name, key in (("HOne", "H1_hard95_vs_base"),
                          ("HTwo", "H2_conf80_vs_base"),
                          ("HThree", "H3_bins10_vs_base")):
            M[f"Abl{tag}{name}"] = contrast(reg, key)

    # ---- calibration-only screen --------------------------------------------
    def pos(s, rho="rho0.1"):
        return scr[s][rho]["hard"]["screen_positive_folds"]
    M["ScrDeployedPos"] = f"{pos('deployed')}/5"
    M["ScrNSevenPos"] = f"{pos('n771')}/5"
    M["ScrNTwoPos"] = f"{pos('n200')}/5"
    M["ScrNFivePos"] = f"{pos('n50')}/5"
    M["ScrDensePos"] = f"{pos('dense')}/5"
    for s, tag in (("deployed", "Deployed"), ("n771", "NSeven"), ("n200", "NTwo"),
                   ("n50", "NFive"), ("dense", "Dense")):
        h = scr[s]["rho0.1"]["hard"]
        M[f"Scr{tag}Auc"] = f"{scr[s]['auc_test_r1']:.3f}"
        M[f"Scr{tag}Calls"] = f"{h['calls']:.3f}"
        M[f"Scr{tag}Net"] = f"{h['net_unweighted']:+.4f}"
        M[f"Scr{tag}CI"] = f"[{h['boot']['lo95']:+.4f}, {h['boot']['hi95']:+.4f}]"
        a = scr[s]["rho0.1"]["all"]
        M[f"Scr{tag}AllNet"] = f"{a['net_unweighted']:+.4f}"
        M[f"Scr{tag}AllCI"] = f"[{a['boot']['lo95']:+.4f}, {a['boot']['hi95']:+.4f}]"

    # ---- gateway stress test ------------------------------------------------
    wiki = gw["wikipedia"]
    M["GwWikiAgree"] = f"{wiki['agreement']:.3f}"
    M["GwWikiDelta"] = f"{wiki['delta_hat']:.3f}"
    M["GwWikiDiff"] = f"{wiki['diff_hat']:+.4f}"
    M["GwWikiCI"] = f"[{wiki['diff_lo']:+.4f}, {wiki['diff_hi']:+.4f}]"
    M["GwCells"] = str(len(gw["cells"]))
    M["GwTostCells"] = str(sum(c["diff"]["equiv"] for c in gw["cells"]))
    M["GwPosCells"] = str(sum(c["diff"]["lo95"] > 0 for c in gw["cells"]))
    zero = [c for c in gw["cells"] if c["h"] == 3 and c["width"] == 5 and c["delta"] == 0.0
            and c["sigma"] == 0.0][0]
    M["GwZeroDiff"] = f3(zero["diff"]["mean"])
    M["GwZeroCI"] = f"[{zero['diff']['lo95']:+.3f}, {zero['diff']['hi95']:+.3f}]"
    M["GwZeroIndex"] = f"{zero['index_mean']:.3f}"
    M["GwZeroMyopic"] = f"{zero['myopic_mean']:.3f}"
    mx = max(gw["cells"], key=lambda c: c["diff"]["mean"])
    M["GwMaxDiff"] = f3(mx["diff"]["mean"])
    M["GwMaxWidth"] = str(mx["width"])
    M["GwMaxSigma"] = f"{mx['sigma']:.2f}"

    # ---- call placement -----------------------------------------------------
    t1 = th["cells"]["trained"]["1"]
    t01 = th["cells"]["trained"]["0.1"]
    z01 = th["cells"]["zeroshot"]["0.1"]
    M["ThRhoOneThompsonCalls"] = f"{t1['thompson']['calls']:.1f}"
    M["ThRhoOneThompsonFetch"] = f"{t1['thompson']['frac_fetched']:.2f}"
    M["ThRhoOneBandCalls"] = f"{t1['band']['calls']:.1f}"
    M["ThRhoOneBandFetch"] = f"{t1['band']['frac_fetched']:.2f}"
    M["ThOursCallsPtOne"] = f"{t01['ours']['calls']:.1f}"
    M["ThOursSkipPtOne"] = f"{t01['ours']['frac_saved']:.2f}"
    M["ThTopmCallsPtOne"] = f"{t01['topm']['calls']:.1f}"
    M["ThTopmFetchPtOne"] = f"{t01['topm']['frac_fetched']:.2f}"
    M["ThOursSkipZero"] = f"{z01['ours']['frac_saved']:.2f}"
    M["ThTopmSkipZero"] = f"{z01['topm']['frac_saved']:.2f}"
    M["ThBandSkipZero"] = f"{z01['band']['frac_saved']:.2f}"

    # ---- cached Prop. 1 curve ----------------------------------------------
    for tag, key in (("Trained", "trained"), ("Weak", "zeroshot")):
        d = pc[key]
        M[f"Pc{tag}Gain"] = f"{d['overall']['mean']:+.4f}"
        M[f"Pc{tag}CI"] = f"[{d['overall']['lo95']:+.4f}, {d['overall']['hi95']:+.4f}]"
        M[f"Pc{tag}Cov"] = f"{d['coverage']*100:.0f}\\%"
        M[f"Pc{tag}N"] = f"{d['n_cached']:,}".replace(",", "{,}")
        ok = [b["bin"] for b in d["bins"] if b["passes"]]
        M[f"Pc{tag}LoBin"] = str(min(ok)) if ok else "--"
        M[f"Pc{tag}HiBin"] = str(max(ok)) if ok else "--"

    # ---- sensitivity --------------------------------------------------------
    M["SensRhoClose"] = f"{0.25:.2f}"
    M["SensLamClose"] = f"{1.5:.1f}"
    M["SensRegionBins"] = str(se["lambda_grid"][4]["bins"])

    lines = ["% generated by artifact/code/make_numbers_v2.py from saved result files",
             "% do not edit by hand"]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, len(M), "macros")


if __name__ == "__main__":
    main()
