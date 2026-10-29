#!/bin/bash
# Test-split runs with the judge, in priority order (non-LLM policies and the index-policy rows come from stage_nollm.sh).
# Resumable: finished crawls are skipped. Run from artifact/code.
set -e; PY=../venv/bin/python; L=../logs; NQ=${NQ:-80}; NS=${NS:-40}; NL=${NL:-25}; B=${B:-30}
par() { N=${NSH:-6}; for s in $(seq 0 $((N-1))); do $PY run_policies.py "$@" --shard $s/$N > $L/test_$(echo "$@" | tr ' /' '__' | cut -c1-60)_$(echo "$@" | cksum | cut -c1-8)_$s.log 2>&1 & done; wait; }
LAM=$($PY -c "import json;print(json.load(open('../data/models/lambda.json'))['25']['lam'])")
TOPM=$(cat ../data/models/topm.json); TS=$(cat ../data/models/thompson.json)
par --split test --policies ours --rho 0.1 --budget $B --limit_q $NQ --params "{\"lam\":$LAM}"
par --split test --policies topm --rho 0.1 --budget $B --limit_q $NQ --params "$TOPM"
par --split test --policies always --rho 0.1 --budget $B --limit_q $NQ
[ -f ../results/gain_curve.json ] || $PY eval_gain_curve.py --lam $LAM --limit_q $NQ > $L/eval_gain_curve.log 2>&1
for r in 1.0 0.5 0.2 0.05; do
  par --split test --policies ours --rho $r --budget $B --limit_q $NS --params "{\"lam\":$LAM}"
  par --split test --policies always --rho $r --budget $B --limit_q $NS
done
echo STRONG_CORE_DONE > $L/pipeline_test_core.done
[ -f $L/weak_train.done ] || ./pipeline_weak.sh
./pipeline_test_weak.sh
par --split test --policies thompson --rho 0.1 --budget $B --limit_q $NQ --params "$TS"
# rho-sweep extras (mostly served from the judge cache of the rho = 0.1 cells)
for r in 1.0 0.5 0.2 0.05; do par --split test --policies topm --rho $r --budget $B --limit_q $NS --params "$TOPM"; done
for r in 1.0 0.5 0.2 0.05; do SF_MODELS=../data/models_weak par --split test --policies always --rho $r --budget $B --limit_q $NS --tag _weak; done
for r in 1.0 0.5 0.2 0.05; do SF_MODELS=../data/models_weak par --split test --policies topm --rho $r --budget $B --limit_q $NS --tag _weak --params "$TOPM"; done
$PY analyze.py > $L/analyze.log 2>&1
for f in 0.5 2; do par --split test --policies ours --rho 0.1 --budget $B --limit_q $NS --tag _lamx$f --params "{\"lam\":$($PY -c "print($LAM*$f)")}"; done
par --split test --policies llm_only --rho 0.1 --budget $B --limit_q $NL
$PY analyze.py > $L/analyze.log 2>&1
echo TEST_DONE > $L/pipeline_test.done
