# Proofs and numeric checks

Notation. A fetch costs `c_f > 0`, an LLM inspection of one link costs `c_l > 0`, and
`rho = c_l / c_f`. The crawl objective is the expected number of target pages fetched under a
total cost budget `B`. `lam >= 0` is the price of one unit of cost (the multiplier of the
budget constraint) and `tau = lam * c_f` is the price of one fetch in units of reward.

## Proposition 1 (escalation interval)

**Setting.** A link has a binary state `Y` (1 if its page is a target) with calibrated prior
`p = P(Y = 1)`. Three actions are available: skip (value 0), fetch blind (cost `c_f`, reward
`Y`), or inspect (cost `c_l`), observe a signal `S`, and then fetch or skip.

**Assumptions.** (A1) Link values are independent across links and the reward is additive.
(A2) The prior `p` is calibrated. (A3) The signal has likelihoods `f_1(s)` and `f_0(s)` given
`Y = 1` and `Y = 0`, and these do not depend on `p` (the signal is conditionally independent
of the encoder score given the state). (A4) `0 < tau < 1`.

**Statement.** Let `q(s) = p f_1(s) / (p f_1(s) + (1-p) f_0(s))` be the posterior and
`W(p) = E[(q(S) - tau)^+]`. The gain of inspecting over the best blind action is
`G(p) = W(p) - (p - tau)^+`. Then

1. `G >= 0`, `G(0) = G(1) = 0`, `G` is nondecreasing on `[0, tau]` and nonincreasing on
   `[tau, 1]`.
2. The set of priors where inspection is optimal at price `lam`,
   `E(c_l) = {p : G(p) > lam c_l}`, is an interval that contains `tau`, or it is empty. It is
   nested: `c_l <= c_l'` implies `E(c_l') ⊆ E(c_l)`.
3. For a perfect signal (`S = Y`), `E = (rho tau / (1 - tau), 1 - rho)`, which is nonempty
   exactly when `rho < 1 - tau`. Its width `1 - rho - rho tau / (1 - tau)` decreases linearly
   in `rho`.
4. Under (A1) the policy that applies the per-link rule (inspect on `E`, fetch blind when
   `p > tau` outside `E`, skip otherwise, and fetch after inspection when `q > tau`) maximises
   `E[reward] - lam E[cost]` over all policies, and hence maximises expected reward among all
   policies whose expected cost does not exceed its own.

**Proof.** Write the marginal density of the signal as `m(s) = p f_1(s) + (1-p) f_0(s)`. Then

    W(p) = ∫ (q(s) - tau)^+ m(s) ds = ∫ ( p (1 - tau) f_1(s) - tau (1 - p) f_0(s) )^+ ds.

For each `s` the integrand is the positive part of a function that is affine in `p`, so it is
convex in `p`, and `W` is convex. `W(0) = 0` and `W(1) = 1 - tau`. By Jensen and
`E[q(S)] = p`, `W(p) >= (p - tau)^+`, so `G >= 0`, and `G(0) = G(1) = 0`.
On `[0, tau]`, `G = W`, which is convex, nonnegative and zero at 0. A convex function that
attains its minimum at the left end of an interval is nondecreasing on it. On `[tau, 1]`,
`G(p) = W(p) - p + tau` is convex, nonnegative and zero at 1, so it attains its minimum at
the right end and is nonincreasing. This proves 1. A function that is nondecreasing up to
`tau` and nonincreasing after it has interval upper level sets containing `tau`, and level
sets shrink as the level `lam c_l` grows, which proves 2.
For a perfect signal `W(p) = p (1 - tau)`. On `p <= tau`, `G(p) = p(1 - tau) > lam c_l =
rho tau` iff `p > rho tau / (1 - tau)`. On `p >= tau`, `G(p) = tau (1 - p) > rho tau` iff
`p < 1 - rho`. The two conditions are compatible iff `rho tau < (1 - rho)(1 - tau)`, i.e.
`rho < 1 - tau`. This proves 3.
For 4, with additive rewards and independent links the Lagrangian
`E[reward] - lam E[cost]` is a sum over links of terms that depend on one link each, so it is
maximised by maximising each term, whose three candidate values are `0`, `p - tau` and
`W(p) - lam c_l`. If a policy `pi` had expected cost at most that of the per-link policy
`pi*` and strictly larger expected reward, its Lagrangian value would be strictly larger,
a contradiction. ∎

**Remarks.** (i) The statement is the classical single-peakedness of the value of
information at the decision threshold, specialised to a two-cost search problem. We claim
the application and its measurable consequences, not the convexity argument. (ii) (A3) is an
idealisation. The experiments therefore estimate the gain curve directly from held-out
links with true labels and report its shape. (iii) Item 4 is optimality for the expected
cost relaxation. Under a hard budget the rule is optimal up to the last, possibly
fractional, item, as in a fractional knapsack.

**How the crawler uses it.** The implemented rule does not trust (A3). For each bin of the
scorer's prior index, `Models.tables` in `code/crawl.py` estimates the gain `G` directly as
the mean realised Lagrangian payoff of the decision taken with the cross-fitted posterior minus
that of the decision taken with the prior, both scored with the true layer of each calibration
link, and escalates in the bins where it exceeds `lam c_l`. Proposition 1 predicts that these
bins form an interval around `tau` that narrows with `rho`; `code/eval_gain_curve.py` measures
the same curve on held-out test links.

**Numeric check.** `code/check_prop1.py` draws 2000 random signal models (2 to 7 outcomes)
and thresholds and verifies items 1 and 2 on a grid of 2001 priors, then verifies the closed
form of item 3 for 2000 random `(tau, rho)` pairs. Result
(`results/theory/prop1_check.json`): zero violations of nonnegativity, of either
monotonicity, of the interval property, of nestedness and of the closed form. The same script
runs a synthetic hard-budget Monte Carlo on a flat frontier (labelled synthetic), where the
interval rule matches inspect-before-every-fetch at small `rho` and matches never-inspect at
`rho >= 0.5`, while inspect-before-every-fetch falls below never-inspect.

## Proposition 2 (max-ratio index on forests)

**Setting.** Pages form a rooted forest. A page becomes available once its parent has been
fetched (roots are available at the start). Page `v` has fetch cost `c_v > 0` and value
`p_v >= 0`, the probability that it is a target.

**Assumptions.** (B1) The forest, costs and values are known and the reward of a set of
fetched pages is additive, `p(S) = sum_{v in S} p_v` (expected recall is additive). (B2) There
is no inspection (Proposition 1 handles inspection of a single link).

**Definitions.** A set `S` is feasible if it is ancestor closed. For a page `v`, let
`T(v)` range over nonempty subtrees rooted at `v` (connected, containing `v`, inside the
subtree of `v`). The index is

    gamma(v) = max_{T(v)} p(T) / c(T).

The index policy always fetches an available page of maximal index. For `lam >= 0` let
`S_lam = {v : gamma(u) > lam for every u on the path from the root of v's tree to v}`.

**Statement.**
1. `S_lam` maximises `p(S) - lam c(S)` over feasible sets.
2. For every `lam` simultaneously, the index policy fetches every page of `S_lam` before any
   page outside `S_lam`.
3. Consequently, for every breakpoint budget `B = c(S_lam)`, the first pages fetched by the
   index policy form a feasible set of maximal value among all feasible sets of cost at
   most `B`.
4. (Gateway instance.) Take unit costs, a decoy page `d` with value `eps`, a gateway page `g`
   with value 0 and a child `t` of `g` with value 1. With budget 2, the myopic rule that
   fetches the available page with the largest `p_v / c_v` collects `eps`, the index policy
   collects 1 (`gamma(g) = 1/2 > eps`), and the ratio `1/eps` is unbounded.

**Proof.** Define `h(v; lam) = max_{T(v)} [p(T) - lam c(T)]`. Splitting a subtree rooted at
`v` into `v` and the subtrees rooted at a subset of its children gives the recursion
`h(v) = p_v - lam c_v + sum_{w child of v} max(0, h(w))`. Moreover `h(v; lam) > 0` iff some
`T(v)` has `p(T) > lam c(T)`, iff `gamma(v) > lam`.
(1) Let `S` be feasible. Decompose `S` into maximal subtrees and compare with `S_lam` top
down. For a root `r`: if `gamma(r) <= lam` then every subtree rooted at `r` has nonpositive
Lagrangian value, so removing the component of `S` at `r` does not decrease the value. If
`gamma(r) > lam` then `h(r) > 0`, and the best subtree rooted at `r` is obtained from the
recursion by including `r` and recursing into exactly the children with `h(w) > 0`, that is
`gamma(w) > lam`. Induction on depth shows that the union over roots of these best subtrees
is `S_lam` and that its value is `sum_r max(0, h(r))`, an upper bound for any feasible `S`.
(2) Suppose the policy fetches a page `v` outside `S_lam` while some page of `S_lam` is
unfetched, and consider the first such moment. Before it, only pages of `S_lam` were
fetched, so the parent of `v` (if any) is in `S_lam` and all proper ancestors of `v` have
index above `lam`. Since `v` is not in `S_lam`, `gamma(v) <= lam`. Let `w` be an unfetched
page of `S_lam` of minimal depth. Its parent is in `S_lam` (ancestor closure) and is fetched
by minimality, so `w` is available, and `gamma(w) > lam >= gamma(v)`. The policy would have
fetched `w`, a contradiction.
(3) By 2 the policy has fetched exactly `S_lam` after `|S_lam|` steps, at cost `c(S_lam)`.
If a feasible `S` with `c(S) <= c(S_lam)` had `p(S) > p(S_lam)` then
`p(S) - lam c(S) > p(S_lam) - lam c(S_lam)`, contradicting 1.
(4) Direct computation. ∎

**Remarks.** (i) The max-ratio index over rooted subtrees is the object used by Horn (1972)
for sequencing with tree precedence, and it is the Gittins index of the deterministic
branching bandit. For random subtrees whose content is revealed by fetching, the index
becomes a supremum over stopping rules of expected reward over expected cost, and the
optimality of index policies for branching bandits is due to Weiss (1988) and Keller and
Oldale (2003). Pandora's box with order constraints is treated by Boodaghians et al. (2020).
We do not claim these results. (ii) In the crawler, the subtree below a link is unknown.
The implemented index is a plug-in chain approximation: with calibrated probabilities
`pi_j` that the nearest target is exactly `j` fetches away through the link (`j = 1..K`),
`R_k = sum_{j<=k} pi_j` and a per-level descent cost `beta` measured on calibration crawls,

    index = max_k R_k / ( sum_{j<=k} pi_j c(j) + (1 - R_k) c(k) ),   c(j) = 1 + beta (j - 1).

This is a modelling choice, not a theorem, and the experiments test it against the myopic
rule (`k = 1` only) and against ranking by `R_K` without cost pricing.

**Numeric check.** `code/check_prop2.py` draws 1500 random forests with 3 to 10 pages,
random costs in `[0.5, 2]` and sparse random values, computes the index by bisection on the
recursion, enumerates all feasible sets, and checks items 1 to 3 at every breakpoint (9887
breakpoints). Result (`results/theory/prop2_check.json`): zero violations. On the gateway
instance the measured ratio is `1/eps` for `eps` in {0.3, 0.1, 0.01, 0.001}.
