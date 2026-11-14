# Artifact manifest: paper macro -> script -> saved result

Regeneration order (all read-only with respect to saved runs, logs and frozen hashes):

1. `../.venv-wsl/bin/python code/analyze.py` -> `manuscript/numbers.tex`; its WSL output is saved as `manuscript/numbers_wsl_regenerated.tex`, which the numbers step reads (the macOS `numbers.tex` is kept for provenance)
2. `../.venv-wsl/bin/python code/round1_confirm.py` -> `results/tables/round1_confirm.json` (confirmatory analysis, blocks 80-200)
3. `../.venv-wsl/bin/python code/inventory_runs.py` -> `results/tables/inventory.json` (per-cell crawl inventory)
4. `../.venv-wsl/bin/python code/gateway_synth.py` -> `results/theory/gateway_synth.json` (exploratory synthetic mechanism check)
5. `../.venv-wsl/bin/python code/make_numbers_and_table.py` -> `manuscript/numbers_all.tex`, `manuscript/sections/tab_r1.tex`
6. `../.venv-wsl/bin/python code/verify_table.py` -> checks every number in `tab_r1.tex` against `round1_confirm.json`
7. `cd manuscript && latexmk -pdf main.tex` -> `manuscript/main.pdf`

`main.tex` inputs exactly one numbers file, `numbers_all.tex`, built from the WSL regeneration
`numbers_wsl_regenerated.tex`. The saved macOS `numbers.tex` (round 0) and `numbers_r1.tex`
(stale, superseded) are kept for provenance and are not input.

| macro | value | origin | producing script | saved result |
|---|---|---|---|---|
| `\AUCEncoder` | 0.974 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\AUCJudge` | 0.925 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\AUCWeakJudge` | 0.966 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\AUCWeakPrior` | 0.950 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\CalHardPct` | 16\% | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\DeltaTOST` | 0.02 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\FetchSecPtOne` | 4.2 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\FullBFS` | 0.287 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\FullBM` | 0.513 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\FullCov` | 0.486 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\FullIndex` | 0.790 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\FullPR` | 0.004 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\GateCommitDepthThree` | 81\% | DIAG | code/diag_index_vs_myopic.py, code/diag_redundancy.py -> code/make_numbers_and_table.py | results/tables/diag_index.json, diag_redundancy.json |
| `\GateNLinks` | 9{,}994 | DIAG | code/diag_index_vs_myopic.py, code/diag_redundancy.py -> code/make_numbers_and_table.py | results/tables/diag_index.json, diag_redundancy.json |
| `\GateRankIndex` | 52.8 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\GateRankMyopic` | 53.7 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\JudgeSecPerCall` | 0.42 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\LinksSeen` | 141 | DIAG | code/diag_index_vs_myopic.py, code/diag_redundancy.py -> code/make_numbers_and_table.py | results/tables/diag_index.json, diag_redundancy.json |
| `\NCalLinks` | 36{,}000 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NCalLinksWeak` | 9{,}600 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NEdges` | 19.26 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NGraphLinks` | 311{,}256 | DIAG | code/diag_index_vs_myopic.py, code/diag_redundancy.py -> code/make_numbers_and_table.py | results/tables/diag_index.json, diag_redundancy.json |
| `\NJudgeCalls` | 125{,}278 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NJudgePromptTok` | 21.3 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NPages` | 5.23 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NQCal` | 400 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NQTest` | 400 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NQTrain` | 771 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NQVal` | 150 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\NTestGoals` | 1200 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\OutDegMean` | 3.68 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\PTwoSameAll` | +0.014 [-0.014, +0.043] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\PTwoSameDFour` | +0.037 [-0.029, +0.104] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\PTwoSameDThree` | +0.008 [-0.037, +0.054] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\PTwoSameDTwo` | -0.004 [-0.025, +0.017] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\PTwoWideDAll` | +0.005 [-0.014, +0.023] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\PTwoWideDThree` | +0.000 [-0.030, +0.030] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RTBandDOne` | -0.051 [-0.075, -0.031] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RTNQPtOne` | 120 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RTOursCallsPtOne` | 5.5 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RTOursDiffCrawlsPtOne` | 6 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RTTSDOne` | -0.121 [-0.150, -0.093] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZBandCallsPtOne` | 59.7 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZBandDPointPtOne` | +0.042 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZBandDPtOne` | +0.042 [+0.010, +0.074] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZBandPHolmPtOne` | 0.033 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursCallsOne` | 4.8 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursCallsPtOne` | 23.3 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursDOne` | -0.017 [-0.036, +0.003] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursDPointOne` | -0.017 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursDPtOne` | +0.011 [-0.013, +0.035] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursMinusBandPHolmPtOne` | 0.112 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursMinusBandPtOne` | -0.031 [-0.061, +0.000] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursMinusTopmPHolmPtOne` | 0.067 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursMinusTopmPtOne` | -0.039 [-0.074, -0.006] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZOursPHolmPtOne` | 0.395 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZTopmCallsPtOne` | 43.4 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZTopmDPointPtOne` | +0.050 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZTopmDPtOne` | +0.050 [+0.019, +0.082] | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\RZTopmPHolmPtOne` | 0.006 | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\SGainHi` | 1.96 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\SGainLo` | 0.36 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\SGainN` | 6{,}000 | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\SameTopFrac` | 94.8\% | R0 | code/analyze.py | results/runs + results/tables + data/models* + results/gain_curve*.json |
| `\SynthDThree` | +0.667 [+0.621, +0.711] | SYN | code/gateway_synth.py -> code/make_numbers_and_table.py | results/theory/gateway_synth.json |
| `\SynthDepthThree` | 3 | SYN | code/gateway_synth.py -> code/make_numbers_and_table.py | results/theory/gateway_synth.json |
| `\SynthIndexThree` | 1.000 | SYN | code/gateway_synth.py -> code/make_numbers_and_table.py | results/theory/gateway_synth.json |
| `\SynthMyopicThree` | 0.333 | SYN | code/gateway_synth.py -> code/make_numbers_and_table.py | results/theory/gateway_synth.json |
| `\SynthTostCells` | 6 | SYN | code/gateway_synth.py -> code/make_numbers_and_table.py | results/theory/gateway_synth.json |
| `\TestHardPct` | 100\% | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |
| `\ValHardPct` | 19\% | R1 | code/round1_confirm.py -> code/make_numbers_and_table.py | results/tables/round1_confirm.json |

Table `tab:main` is generated by `code/make_numbers_and_table.py` from the same
`results/tables/round1_confirm.json`; its cells are the macros above with the `RT`/`RZ` prefixes.

`SYN` macros come from the exploratory synthetic mechanism check of `code/gateway_synth.py`
(54 cells on a synthetic tree). They are not part of Table 1 and enter no confirmatory comparison.

Used but not defined in numbers_all.tex: Bbbk, Delta

## Round-0 values that changed when regenerated on WSL (and why)

Four macros differ between the saved `manuscript/numbers.tex` (macOS) and the file regenerated
on WSL by `code/analyze.py`. All four are explained by the round-1 refit of the posterior model
(ROUND1_PLAN change 1: prior calibrators cross-fitted by question) and by the growth of the judge
cache, not by platform drift. No recall, interval, p-value or TOST verdict changed.

| macro | saved (pre-session) | regenerated on WSL | cause |
|---|---|---|---|
| `AUCEncoder` | 0.976 | 0.974 | frozen `posterior.pkl` uses the cross-fitted prior; the saved file came from the pre-refit fit |
| `AUCPost` | 0.978 | 0.977 | same refit |
| `AUCWeakPrior` | 0.951 | 0.950 | same refit, zero-shot scorer |
| `NJudgeCalls` / `NJudgePromptTok` | 76,246 / 13.0 | 125,278 / 21.3 | the LLM cache gained the confirmatory calls |

AUC recomputed read-only from the saved posteriors (layer-1 target, this script):

| model file | n | AUC prior | AUC posterior (out-of-fold) | has cross-fitted prior |
|---|---:|---:|---:|---|
| `data/models/posterior.pkl` | 36000 | 0.97422 | 0.97672 | yes |
| `data/models/posterior_round0.pkl` | 36000 | 0.97589 | 0.97771 | no |
| `data/models_weak/posterior.pkl` | 9600 | 0.94988 | 0.97834 | yes |
| `data/models_weak/posterior_round0.pkl` | 9600 | 0.95083 | 0.97850 | no |

`data/models/posterior_info.json` (frozen) reports auc_prior 0.97422 and auc_post_oof 0.97672, matching `posterior.pkl`.
`posterior_info_round0.json` reports 0.97589 and 0.97771, matching `posterior_round0.pkl`
and the saved numbers.tex (0.976 / 0.978). The frozen pickle hash is the one listed in
`preregistration/round1_rule_hashes.txt`.

`NJudgeCalls` is the number of rows in `data/cache/llm.sqlite` (`count(*)` over table `insp`):
distinct (model, question, parent, target) judgements cached across every run in the artifact,
exploration and confirmatory, reused by all policies. It is not the number of policy-level calls
(those are the per-crawl call means in Table 1). `NJudgePromptTok` is the sum of the prompt tokens
of those distinct judgements. `code/judge_usage_snapshot.py` records the final count read-only in
`results/llm_usage_final.json` (cache sha256 in that file); the earlier `results/llm_usage.json`
(76,246 rows) was written before the confirmatory crawls and is kept unchanged.

The confirmatory table has no saved predecessor: `numbers_r1.tex` (kept unchanged) never
contained cell macros, and `results/tables/round1.json` has empty cell blocks. The table is
therefore computed anew from the saved run files of blocks 80-199, not a re-run of a saved value.

## Beta regeneration caveat

`data/models/calib.pkl` stores beta = 2.5846579129. Refitting `code/fit_calib.py index_B` on the saved
`data/rows/cal_index_B_s*.pkl` (byte-identical to the 2026-10-29 backup) gives
2.5473 with the default `numpy.argsort` and 2.5660 with a stable sort;
n_links = 310,397 and n_pages_for_beta = 1447 match the
stored fit, and the isotonic curves agree at the fitted quantiles. The gap is tie-breaking in
the rank statistic (many links share the same index), which is platform-dependent, not row drift.
All saved runs and every paper number use the frozen beta and are unaffected; off-platform
regeneration will give a slightly different beta unless the original platform's sort order is
reproduced. Diagnostic: `code/level_shift.py` -> `results/exploratory/level_shift.json`.
