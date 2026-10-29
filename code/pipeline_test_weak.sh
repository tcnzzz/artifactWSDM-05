#!/bin/bash
# Test-split runs of the weak-scorer regime (after pipeline_weak.sh). Hyperparameters of top-m and Thompson are those chosen on
# validation for the strong regime. Run from artifact/code.
set -e; PY=../venv/bin/python; L=../logs; NQ=${NQW:-50}; NS=${NSW:-30}; B=${B:-30}; export SF_MODELS=../data/models_weak
par() { N=${NSH:-6}; for s in $(seq 0 $((N-1))); do $PY run_policies.py "$@" --shard $s/$N > $L/testweak_$(echo "$@" | tr ' /' '__' | cut -c1-60)_$(echo "$@" | cksum | cut -c1-8)_$s.log 2>&1 & done; wait; }
LAM=$($PY -c "import json;print(json.load(open('../data/models_weak/lambda.json'))['25']['lam'])")
TOPM=$(cat ../data/models/topm.json); TS=$(cat ../data/models/thompson.json)
par --split test --policies index --rho 0.1 --budget 50 --tag _weak
par --split test --collect index --budget 50 --tag _weakT --no_path
par --split test --policies ours --rho 0.1 --budget $B --limit_q $NQ --tag _weak --params "{\"lam\":$LAM}"
par --split test --policies topm --rho 0.1 --budget $B --limit_q $NQ --tag _weak --params "$TOPM"
par --split test --policies always --rho 0.1 --budget $B --limit_q $NQ --tag _weak
$PY eval_gain_curve.py --lam $LAM --limit_q $NQ --per_crawl 12 --rows index_weakT --out_tag _weak > $L/weak_eval_gain_curve.log 2>&1
for r in 1.0 0.5 0.2 0.05; do par --split test --policies ours --rho $r --budget $B --limit_q $NS --tag _weak --params "{\"lam\":$LAM}"; done
par --split test --policies thompson --rho 0.1 --budget $B --limit_q $NS --tag _weak --params "$TS"
echo WEAKTEST_DONE > $L/pipeline_test_weak.done
