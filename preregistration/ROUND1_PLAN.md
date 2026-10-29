# Round-1 confirmatory evaluation: frozen rule and analysis plan

Written 2026-10-06 22:35 +0700, before any validation run of the rule below and before any new test
run. Hashes of the code and fitted models at freeze time are in `round1_rule_hashes.txt`. The
baseline configurations chosen on validation are appended to `round1_baselines.json` (with its
own hash and timestamp) before the first confirmatory test run.

## Chronology of earlier work (exploration, superseded)
- 2026-10-06 07:15 to 19:22: point-estimate rule (`ours`) run on the first 80 test questions (trained
  scorer, rho sweep on the first 40), top-m, judge-before-every-fetch, Thompson over links, LLM-only.
- 19:18 to 20:19: zero-shot runs of the point-estimate rule and top-m on the first 50 test questions.
- 20:30: after those test runs existed, the rule was changed to a conservative region (95% one-sided
  lower bound plus the single run of passing bins containing the point-estimate peak) and run on
  the same test questions (`ours_cons*`). That change was made with test results of the alternative in
  hand. All runs on test questions 1 to 80 are reported only as superseded exploration.

## Changes fixed now, before any new run
1. Prior calibrators cross-fitted by question (5 folds) when estimating the gain table, like the
   posterior (`fit_posterior.py --calib_rows`). The table gain is the chain-model payoff of acting on
   the out-of-fold posterior minus that of acting on the out-of-fold prior, scored with true layers.
2. Final rule ("TwoCost"): a prior-index bin escalates if and only if the one-sided 95% lower
   bootstrap bound of its gain (1000 resamples over calibration questions, seed 0) exceeds
   lambda * rho. No contiguity step. (`Models.tables(..., conservative="all")`.) lambda is the
   validation price at budget 25 (unchanged). Everything else as in round 0.
3. Runs use budget B = 25 exactly (no truncation of B = 30 runs).

## Confirmatory set
Test questions at positions 80 to 399 of `data/goals_test.jsonl` (first-appearance order): 320
questions, 960 crawls per policy and regime. None of them has been run with any judge policy, and the
judge cache holds no call for them.

## Policies (both regimes: trained scorer, zero-shot scorer)
scorer alone (index), TwoCost (rule above), top-m plus uncertainty, Thompson sampling over links,
judge-band cascade (call iff prior index in [a lambda, b lambda], same posterior and ranking as
TwoCost), all at rho in {0.1, 0.5, 1.0}; judge before every fetch at rho = 0.1 only.

## Baseline tuning (validation questions at positions 20 to 69, 150 crawls, never test)
Per regime and per rho: top-m (m, m_u) in {0,1,3,5} x {0,1,2}; Thompson kappa in {1,3,10,30};
band (a, b) with a in {0.25,0.5,1}, b in {1,2,4,inf}, a < b. Selection: highest mean recall at cost 25;
ties within 0.001 go to fewer judge calls. TwoCost is run on the same validation questions for the
record only; it is not tuned.

## Analysis
- Outcome: recall of the two gold pages at cost 25.
- Primary family per regime, rho = 0.1: each judge policy minus scorer alone (4 contrasts) and
  TwoCost minus each tuned baseline (3 contrasts).
- Inference: paired bootstrap over questions, 10,000 resamples, seed 0. Superiority: 95% percentile
  interval and bootstrap p-value, Holm-adjusted within each family of 4 (vs scorer alone) and 3
  (TwoCost vs baselines). Equivalence: TOST with margin delta = 0.02 recall, declared equivalent when
  the 90% interval lies inside (-0.02, +0.02).
- Secondary: the same contrasts at rho = 0.5 and 1.0, judge calls per crawl, depth strata.
- Prediction 2 (no judge involved): index minus same-head myopic rule, per depth, on the 960
  confirmatory crawls, with TOST at delta = 0.02.
- Redundancy: for gold pages that a judge policy fetched after judging their link, the fraction also
  fetched by the scorer-alone crawl of the same goal.

## Amendment 1 (2026-10-06 22:53 +0700), before any validation or test result was inspected
The judge runs at about 1.5 to 2.5 calls per second on the shared laptop GPU, so the scope is reduced
for compute: rho in {0.1, 1.0} only (0.5 dropped); grids top-m (m, m_u) in {0,1,3,5} x {0,1},
Thompson kappa in {3, 10, 30}, band a in {0.25, 0.5, 1} and b in {2, inf}; judge-before-every-fetch
dropped from the confirmatory set (it remains in the exploration results). The TwoCost rule, the
confirmatory questions, the margin and the tests are unchanged. The primary family vs scorer alone
keeps its 4 contrasts (TwoCost, band, top-m, Thompson).

## Amendment 2 (written before any confirmatory test run)
Measured judge throughput on the shared laptop GPU is about 1 to 1.6 calls per second, too slow for
all 320 confirmatory questions. The confirmatory questions are therefore run in file order in blocks
of 40 (80-119, 120-159, ...), each block covering every cell. No new block starts after
2026-10-07 16:00 +0700. The analysis uses all completed blocks. The stopping time is fixed by the
clock, not by results, and no confirmatory result is inspected before it. If fewer than 320 questions
are reached, the paper reports how many.
