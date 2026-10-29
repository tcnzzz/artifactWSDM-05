"""Numeric check of Proposition 1 (escalation interval).

Binary link state Y with calibrated prior p. Fetch: cost c_f, reward Y. Inspection: cost c_l,
signal S with likelihoods f1 (Y=1), f0 (Y=0) that do not depend on p. Budget price lam,
tau = lam*c_f. Gain of inspecting over the best blind action:
    G(p) = E[(q(S)-tau)^+] - max(p-tau, 0).
Claims checked on random instances:
 (a) G >= 0, G(0)=G(1)=0, nondecreasing on [0,tau], nonincreasing on [tau,1];
 (b) {p : G(p) > lam*c_l} is an interval containing tau (or empty), nested in c_l;
 (c) perfect signal: the set equals (rho*tau/(1-tau), 1-rho), rho=c_l/c_f, empty iff rho>=1-tau;
 (d) hard-budget Monte Carlo (synthetic): the interval rule beats always/never inspect.
Writes results/theory/prop1_check.json.
"""
import json, os, numpy as np
rng = np.random.default_rng(20261005)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "theory"); os.makedirs(OUT, exist_ok=True)

def gain(p, f1, f0, tau):
    # p: (P,), f1,f0: (m,)
    num = p[:, None] * f1[None, :]; den = num + (1 - p[:, None]) * f0[None, :]
    W = np.maximum(num - tau * den, 0).sum(1)          # E[(q-tau)^+] = sum_s max(p f1 - tau*den, 0)
    return W - np.maximum(p - tau, 0)

viol = dict(nonneg=0, mono_left=0, mono_right=0, interval=0, nested=0, closed_form=0)
n_inst = 2000; grid = np.linspace(0, 1, 2001)
for _ in range(n_inst):
    m = rng.integers(2, 8)
    f1 = rng.dirichlet(np.ones(m) * rng.uniform(0.3, 3)); f0 = rng.dirichlet(np.ones(m) * rng.uniform(0.3, 3))
    tau = rng.uniform(0.01, 0.95)
    g = gain(grid, f1, f0, tau)
    viol["nonneg"] += bool((g < -1e-10).any())
    left = grid <= tau; right = grid >= tau
    viol["mono_left"] += bool((np.diff(g[left]) < -1e-10).any())
    viol["mono_right"] += bool((np.diff(g[right]) > 1e-10).any())
    prev = None
    for thr in np.sort(rng.uniform(0, g.max() + 1e-9, 4)):
        idx = np.where(g > thr)[0]
        if len(idx):
            viol["interval"] += bool((np.diff(idx) != 1).any())
            cur = (idx[0], idx[-1])
            if prev is not None: viol["nested"] += not (cur[0] >= prev[0] and cur[1] <= prev[1])
            prev = cur
# (c) perfect signal closed form
for _ in range(2000):
    tau = rng.uniform(0.01, 0.95); rho = rng.uniform(0.01, 1.2)
    g = gain(grid, np.array([1.0, 0.0]), np.array([0.0, 1.0]), tau)
    idx = np.where(g > tau * rho + 1e-12)[0]          # lam*c_l = tau*rho
    lo, hi = rho * tau / (1 - tau), 1 - rho
    if rho >= 1 - tau: viol["closed_form"] += len(idx) > 0
    else:
        viol["closed_form"] += not (abs(grid[idx[0]] - lo) < 2e-3 and abs(grid[idx[-1]] - hi) < 2e-3)

# (d) synthetic hard-budget Monte Carlo on a flat frontier
def mc(rho, B=30.0, N=400, reps=300, acc=0.85):
    """Flat synthetic frontier, hard budget. The price is chosen per rule by its mean over
    repetitions (never per realisation), so no rule benefits from hindsight."""
    f1 = np.array([acc, 1 - acc]); f0 = np.array([1 - acc, acc]); lams = np.geomspace(0.005, 0.9, 40)
    tot = {"never": 0.0, "always": np.zeros(len(lams)), "interval": np.zeros(len(lams))}
    for r in range(reps):
        p = rng.beta(0.5, 6, N); y = rng.random(N) < p
        s = np.where(y, rng.random(N) < acc, rng.random(N) < 1 - acc)
        q = np.where(s, p * acc / (p * acc + (1 - p) * (1 - acc)), p * (1 - acc) / (p * (1 - acc) + (1 - p) * acc))
        tot["never"] += y[np.argsort(-p)[:int(B)]].sum()
        for mode in ["always", "interval"]:
            for j, lam in enumerate(lams):
                insp = np.ones(N, bool) if mode == "always" else gain(p, f1, f0, lam) > lam * rho
                fetch = np.where(insp, q > lam, p > lam)
                cost = 0.0; got = 0
                for i in np.argsort(-p):                    # the signal is seen only after paying
                    if not (insp[i] or fetch[i]): continue
                    if insp[i]:
                        if cost + rho > B: break
                        cost += rho
                    if fetch[i]:
                        if cost + 1 > B: break
                        cost += 1; got += y[i]
                tot[mode][j] += got
    return {"never": tot["never"] / reps, "always": float(tot["always"].max() / reps), "interval": float(tot["interval"].max() / reps)}
mcres = {str(rho): mc(rho) for rho in [0.05, 0.2, 0.5, 1.0]}
out = dict(n_instances=n_inst, violations=viol, synthetic_budget_mc=mcres)
json.dump(out, open(os.path.join(OUT, "prop1_check.json"), "w"), indent=1); print(json.dumps(out, indent=1))
