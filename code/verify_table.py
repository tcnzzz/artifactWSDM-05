"""Verify every number printed in sections/tab_r1.tex against results/tables/round1_confirm.json."""
import json
import os
import re

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ART, "..", "manuscript")
conf = json.load(open(os.path.join(ART, "results", "tables", "round1_confirm.json")))
tab = open(os.path.join(MS, "sections", "tab_r1.tex")).read().splitlines()

PN = {"TwoCost": "ours", "Band cascade": "band", "Top-$m$ + uncertainty": "topm", "Thompson over links": "thompson"}
RW = {"0.1": "PtOne", "1": "One"}
cur = None
checked = 0
ok = True
for line in tab:
    if "trained scorer:" in line:
        cur = "trained"
    elif "zero-shot scorer:" in line:
        cur = "zeroshot"
    for name, key in PN.items():
        if line.startswith(name) or line.startswith("\\textbf{\\system}"):
            pass
    if cur and (line.startswith("\\textbf{\\system}") or any(line.startswith(n) for n in PN)):
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        pol = "ours" if line.startswith("\\textbf{\\system}") else next(v for k, v in PN.items() if line.startswith(k))
        c1, c2 = conf["cells"][cur]["0.1"][pol], conf["cells"][cur]["1"][pol]
        want = [
            f"{c1['recall']:.3f}",
            None,  # CI string checked separately
            (lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")(c1["vs_scorer"]["p_holm"]),
            f"{c1['calls']:.1f}",
            "eq" if c1["vs_scorer"]["equiv"] else "--",
        ]
        got = [cells[1], cells[2], cells[3], cells[4], cells[5]]
        if got[0] != want[0] or got[2] != want[2] or got[3] != want[3] or got[4] != want[4]:
            print("MISMATCH", cur, pol, got, want)
            ok = False
        d = c1["vs_scorer"]
        ci = f"{d['mean']:+.3f} [{d['lo95']:+.3f}, {d['hi95']:+.3f}]".replace("+0.000", "+0.000")
        if got[1] != ci:
            print("CI MISMATCH", cur, pol, got[1], ci)
            ok = False
        d2 = c2["vs_scorer"]
        ci2 = f"{d2['mean']:+.3f} [{d2['lo95']:+.3f}, {d2['hi95']:+.3f}]"
        if cells[6] != ci2 or cells[7] != f"{c2['calls']:.1f}":
            print("RHO1 MISMATCH", cur, pol, cells[6:8], ci2, f"{c2['calls']:.1f}")
            ok = False
        checked += 1
print(f"checked {checked} policy rows: {'ALL MATCH' if ok else 'ERRORS'}")
# panel headers
for reg in ("trained", "zeroshot"):
    c = conf["cells"][reg]["0.1"]
    print(reg, "nq", c["nq"], "n", c["n"])
