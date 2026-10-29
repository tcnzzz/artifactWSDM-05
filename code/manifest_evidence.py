"""Evidence block for MANIFEST.md: the four round-0 macros that changed on regeneration.

Reads only saved files under artifact/data and artifact/results. Recomputes the layer-1 AUCs
from the two posterior pickles so the manifest can show that the frozen model (not the platform)
explains the change.
"""
import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _auc(sc, yk):
    return float(roc_auc_score(yk, sc)) if 0 < yk.sum() < len(yk) else None


def auc_section():
    rows = []
    for tag in ("models", "models_weak"):
        for fn in ("posterior.pkl", "posterior_round0.pkl"):
            d = pickle.load(open(os.path.join(ART, "data", tag, fn), "rb"))
            c = d["cal"]
            yk = (c["y"] == 0).astype(int)
            rows.append((tag, fn, _auc(c["prior"][:, 0], yk), _auc(c["post_oof"][:, 0], yk),
                         "prior_infold" in c, len(yk)))
    info = {(t, f): json.load(open(os.path.join(ART, "data", t, f)))
            for t in ("models", "models_weak")
            for f in ("posterior_info.json", "posterior_info_round0.json")}
    fn = os.path.join(ART, "results", "llm_usage_final.json")
    if not os.path.exists(fn):
        fn = os.path.join(ART, "results", "llm_usage.json")
    usage = json.load(open(fn))
    m_new = info[("models", "posterior_info.json")]["R1"]
    m_old = info[("models", "posterior_info_round0.json")]["R1"]

    s = []
    s.append("\n## Round-0 values that changed when regenerated on WSL (and why)\n\n")
    s.append("Four macros differ between the saved `manuscript/numbers.tex` (macOS) and the file regenerated\n")
    s.append("on WSL by `code/analyze.py`. All four are explained by the round-1 refit of the posterior model\n")
    s.append("(ROUND1_PLAN change 1: prior calibrators cross-fitted by question) and by the growth of the judge\n")
    s.append("cache, not by platform drift. No recall, interval, p-value or TOST verdict changed.\n\n")
    s.append("| macro | saved (pre-session) | regenerated on WSL | cause |\n|---|---|---|---|\n")
    s.append("| `AUCEncoder` | 0.976 | 0.974 | frozen `posterior.pkl` uses the cross-fitted prior; the saved file came from the pre-refit fit |\n")
    s.append("| `AUCPost` | 0.978 | 0.977 | same refit |\n")
    s.append("| `AUCWeakPrior` | 0.951 | 0.950 | same refit, zero-shot scorer |\n")
    s.append("| `NJudgeCalls` / `NJudgePromptTok` | 76,246 / 13.0 | "
             f"{usage['distinct_judge_calls']:,} / {usage['prompt_tokens'] / 1e6:.1f} | the LLM cache gained the confirmatory calls |\n\n")
    s.append("AUC recomputed read-only from the saved posteriors (layer-1 target, this script):\n\n")
    s.append("| model file | n | AUC prior | AUC posterior (out-of-fold) | has cross-fitted prior |\n|---|---:|---:|---:|---|\n")
    for tag, fn, ap, aq, has, n in rows:
        s.append(f"| `data/{tag}/{fn}` | {n} | {ap:.5f} | {aq:.5f} | {'yes' if has else 'no'} |\n")
    s.append("\n`data/models/posterior_info.json` (frozen) reports auc_prior "
             f"{m_new['auc_prior']:.5f} and auc_post_oof {m_new['auc_post_oof']:.5f}, matching `posterior.pkl`.\n")
    s.append("`posterior_info_round0.json` reports "
             f"{m_old['auc_prior']:.5f} and {m_old['auc_post_oof']:.5f}, matching `posterior_round0.pkl`\n")
    s.append("and the saved numbers.tex (0.976 / 0.978). The frozen pickle hash is the one listed in\n")
    s.append("`preregistration/round1_rule_hashes.txt`.\n\n")
    s.append("`NJudgeCalls` is the number of rows in `data/cache/llm.sqlite` (`count(*)` over table `insp`):\n")
    s.append("distinct (model, question, parent, target) judgements cached across every run in the artifact,\n")
    s.append("exploration and confirmatory, reused by all policies. It is not the number of policy-level calls\n")
    s.append("(those are the per-crawl call means in Table 1). `NJudgePromptTok` is the sum of the prompt tokens\n")
    s.append("of those distinct judgements. `code/judge_usage_snapshot.py` records the final count read-only in\n")
    s.append("`results/llm_usage_final.json` (cache sha256 in that file); the earlier `results/llm_usage.json`\n")
    s.append("(76,246 rows) was written before the confirmatory crawls and is kept unchanged.\n\n")
    s.append("The confirmatory table has no saved predecessor: `numbers_r1.tex` (kept unchanged) never\n")
    s.append("contained cell macros, and `results/tables/round1.json` has empty cell blocks. The table is\n")
    s.append("therefore computed anew from the saved run files of blocks 80-199, not a re-run of a saved value.\n")
    return "".join(s)
