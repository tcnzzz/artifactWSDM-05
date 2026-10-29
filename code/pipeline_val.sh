#!/bin/bash
# Validation stage: budget price (already measured by stage_nollm.sh) and baseline hyperparameters at the headline budget.
set -e; PY=../venv/bin/python; L=../logs; NV=${NV:-20}; B=${B:-25}
par() { for s in 0 1 2; do $PY run_policies.py "$@" --shard $s/3 > $L/val_$(echo "$@" | tr ' /' '__' | cut -c1-70)_$s.log 2>&1 & done; wait; }
[ -f ../data/models/lambda.json ] || { par --split val --policies index --rho 0.1 --budget 50; $PY estimate_lambda.py > $L/estimate_lambda.log 2>&1; }
LAM=$($PY -c "import json;print(json.load(open('../data/models/lambda.json'))['$B']['lam'])")
for c in "1 0" "3 1" "5 2"; do set -- $c; par --split val --policies topm --rho 0.1 --budget $B --limit_q $NV --tag _m$1u$2 --params "{\"m\":$1,\"mu\":$2}"; done
for k in 1 3 10; do par --split val --policies thompson --rho 0.1 --budget $B --limit_q $NV --tag _k$k --params "{\"ts_kappa\":$k}"; done
par --split val --policies ours,always --rho 0.1 --budget $B --limit_q $NV --params "{\"lam\":$LAM}"
$PY select_hparams.py > $L/select_hparams.log 2>&1
echo VAL_DONE > $L/pipeline_val.done
