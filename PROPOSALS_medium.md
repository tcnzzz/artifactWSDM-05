# Medium-priority follow-ups (internal; NOT part of the anonymous supplement)

Nothing here was run for this submission. Each item fixes its success and failure criteria before
any data are collected, so a future run cannot be re-narrated after the fact. Items (a)-(d) map
one-to-one onto the limitations in Section 5. Costs use the measured 7B judge throughput, 0.42 s
per call (`results/judge_latency.json`); a simulation crawl costs seconds of CPU.

**Why the tuned band is the strongest baseline.** The band cascade searches the same band family
as our rule (width `a` in {0.25, 0.5, 1}, threshold `b` in {2, inf}) on validation, with the same
posterior, price lambda and judge; top-m and Thompson are tuned over their own grids on the same
split. Every comparison is at equal realised cost against a tuned baseline, and the learned router
(d) is the one family that could dominate the band, so it is proposed explicitly.

## (a) Synthetic graph with planted gateways
**Question.** On Wikipedia the index and the myopic rule pick the same top link on 94.8% of links
and the index ranks gateway links better on only 44.9% (`diag_index.json`), so the null may be a
property of the graph. Does the index beat the same-head myopic rule when the difference is
planted by construction?
**Design.** Layered DAGs (depth 2-4), 200 and 400 goals, one gold chain per goal. At each
non-terminal layer, one gateway link leads into the gold continuation and several decoys have the
same `pi_1` and the same scorer features but different continuation value. Three generator
settings make the index prefer a non-myopic top link on 20%, 50% and 80% of links. Run the frozen
rule, the binary-head myopic rule, top-m, band, Thompson and the scorer alone; paired bootstrap
over goals (10,000 resamples, seed 0), TOST at 0.02.
**Criteria.** Success: at the 50% and 80% settings the index minus myopic is at least +0.02 recall
with the 95% interval excluding 0 at equal cost, growing with the planted share; equivalence at
20% is the manipulation check. Failure: TOST equivalence at 80%, which would say the index cannot
exploit gateways even when they exist.
**Cost / needs.** CPU minutes; no judge, no laptop GPU. About 1-2 days to write the generator and
reuse `crawl.py` policies on a synthetic graph.
**Label.** Exploratory, outside the confirmatory family: it tests a proposition, not the method on
the paper's data.
**Risks.** The generator can be tuned until the answer is known in advance, so the three shares are
fixed here. Equal features by construction must be verified. The result cannot replace the
Wikipedia null; it can only show the null is graph-driven.
**Status (2026-10-30).** Run as `code/gateway_synth.py`; output `results/theory/gateway_synth.json`.
The implemented design perturbs the one-hop score (`delta`) and the multi-hop layers (`sigma`) and
varies decoy width, rather than the planted-share settings above, and adds a depth-4 boundary cell.
The paper reports one exploratory sentence in Section 2; the full 54-cell grid stays in the artifact.

## (b) Stronger (14B) or listwise judge on test block 80-119
**Question.** All results use one 7B pointwise judge. Does a stronger or listwise judge change the
value of a judge call, the policy ranking, or the conclusion that the judge adds nothing
measurable with the trained scorer?
**Design.** Re-judge the links actually inspected in the eight reported cells at rho = 0.1 on test
questions 80-119 (40 questions, 120 crawls per cell): 9,324 distinct (question, target) links,
counted from the saved `I` events (3,251 trained, 8,032 zero-shot, union 9,324), about 2,500
listwise calls at out-degree 3.68. Arms: same pointwise prompt with a 14B model; listwise, one
call per fetched page, with the 14B and the 7B. Keep scorer, lambda and tuning fixed; analyse the
Table 1 contrasts on this block with the same bootstrap and Holm family.
**Criteria.** Judge quality matters if the judge-gain curve moves by at least 0.02 lambda at the
median decile, or the TwoCost-minus-scorer sign at rho = 0.1 changes with a 95% interval excluding
0. It does not matter if that contrast stays inside (-0.02, +0.02) under TOST in both arms, which
would strengthen the bounded null. Anything else is inconclusive; no arm enters Table 1.
**Cost / needs.** 9,324 calls at 7B and 14B, or ~2,500 listwise calls with 3.7x longer prompts:
about 1.1 h of GPU at 7B, 3-3.5 h at 14B on this machine; no cache sharing between models. Needs
the GPU judge and the laptop; no server.
**Label.** Exploratory. It revisits a confirmatory block with a different judge, which the frozen
plan does not cover; confirmatory status would need Amendment 3.
**Risks.** Forty questions give wide intervals and an inconclusive result is the most likely one.
A 14B model may require different quantization, confounding "stronger" with "different serving";
listwise confounds signal quality with price. One block does not generalise to blocks 120-199.

## (c) Level-matched recalibration
**Question.** The scorer is calibrated where 16% of goals are hard, but the test split is 100% hard
(calibration 65/400, validation 29/150). Does recalibrating on hard goals only close the
predicted-versus-realised gap in judge value, and does it change what the rule does?
**Design.** Refit isotonic calibration and the posterior prior on the 65 hard calibration
questions only (never test or validation), keep encoder and judge fixed, re-estimate lambda on hard
validation questions with the paper's procedure. Re-run the frozen rule and the tuned band on the
120 confirmatory questions at rho = 0.1 and 1.0; compare predicted versus realised judge calls per
crawl, and TwoCost minus scorer and minus band at equal cost, paired over questions, same TOST and
Holm machinery. The paper's confirmatory numbers are not replaced.
**Criteria.** Success: predicted/realised calls move toward 1 by at least 25% relative to the
current calibration, and the recalibrated rule stays within 0.02 recall of the scorer while the
zero-shot gap closes. Failure: the ratio is unchanged, or the recall loss exceeds the margin.
**Cost / needs.** Seconds to recalibrate; 120 x 3 depths x 2 policies x 2 rho = 1,440 new crawls,
bounded by the band's 18.3 calls per crawl with most links cached, under 1 h of GPU. Laptop only.
**Label.** Exploratory as an analysis; adopting a recalibrated rule as the reported method would
need Amendment 3, because rule and calibration are frozen.
**Risks.** 65 calibration questions make isotonic fitting noisy, so a null is as easy to explain as
a shift. Validation-hard is used both for lambda and for checks, and that overlap must be reported.
The level labels come from goal construction and may not capture every kind of test hardness.

## (d) Learned cost-aware router
**Question.** The rule uses a fixed band on the posterior, and the tuned band is the best member of
that family. Can a learned router over scorer uncertainty, frontier size, depth and rho beat the
tuned band at equal realised cost?
**Design.** Features at each decision: posterior mean and dispersion, frontier size, depth,
remaining budget fraction, rho. Train a small LightGBM policy on exploration crawls of train and
calibration splits only, label = realised value of a judge call (whether judging changes the
fetched set or finds gold), cross-fitted by question. Evaluate on validation against the tuned band
at equal cost; only a validation win authorises a run on the 120 confirmatory questions, with the
same bootstrap, TOST and Holm.
**Criteria.** Success: router minus band at least +0.02 recall with the 95% interval excluding 0 on
validation and then on test. Failure: TOST equivalence on validation, in which case the tuned band
stays the method and the router is reported as not better. No test-set tuning.
**Cost / needs.** Minutes to train; 720 crawls on test if it gets there, judge calls capped at 18.3
per crawl with most cached, about 1-1.5 h of GPU. Laptop plus judge cache; no new judge.
**Label.** Exploratory on validation; needs Amendment 3 before any test number enters the
confirmatory table or the rule is replaced.
**Risks.** The value label is observed only for links the exploration policies judged, so training
is off-policy and confounded by depth and by the collecting policy. A router can win on validation
by exploiting the tuning split; cross-fitting limits but does not remove this. Even a validation
win would not establish a test-distribution gain.

**Ordering.** (a) first: cheapest, no judge, tests the proposition directly. (b) and (c) share the
block 80-119 re-judging and rerun infrastructure; (b) tests the judge assumption, (c) the
calibration shift. (d) last: it is the only item that could change the method and needs the most
protection against tuning on the evaluation split.
