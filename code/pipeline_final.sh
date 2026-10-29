#!/bin/bash
# Remaining test cells in priority order (resumable; finished crawls are skipped). Run from artifact/code.
# "_cons" = conservative escalation region (lower confidence bound + single interval), fixed on 2026-10-06 20:30 from
# calibration data only, after the point-estimate region was seen to escalate on noise in the zero-shot regime.
set -e; cd "$(dirname "$0")"; PY=../venv/bin/python; L=../logs; NQ=80; NQW=50; NS=40; NSW=30; NL=25; B=30
par() { N=${NSH:-6}; for s in $(seq 0 $((N-1))); do $PY run_policies.py "$@" --shard $s/$N > $L/final_$(echo "$@" | tr ' /' '__' | cut -c1-60)_$(echo "$@" | cksum | cut -c1-8)_$s.log 2>&1 & done; wait; }
LAM=$($PY -c "import json;print(json.load(open('../data/models/lambda.json'))['25']['lam'])")
LAMW=$($PY -c "import json;print(json.load(open('../data/models_weak/lambda.json'))['25']['lam'])")
TOPM=$(cat ../data/models/topm.json); TS=$(cat ../data/models/thompson.json)
W() { SF_MODELS=../data/models_weak "$@"; }
W par --split test --policies always --rho 0.1 --budget $B --limit_q $NQW --tag _weak
W par --split test --policies ours --rho 0.1 --budget $B --limit_q $NQW --tag _cons_weak --params "{\"lam\":$LAMW,\"conservative\":1}"
par --split test --policies ours --rho 0.1 --budget $B --limit_q $NQ --tag _cons --params "{\"lam\":$LAM,\"conservative\":1}"
[ -f ../results/gain_curve_weak.json ] || W $PY eval_gain_curve.py --lam $LAMW --limit_q $NQW --per_crawl 12 --rows index_weakT --out_tag _weak > $L/weak_eval_gain_curve.log 2>&1
for r in 1.0 0.5 0.2 0.05; do W par --split test --policies ours --rho $r --budget $B --limit_q $NSW --tag _cons_weak --params "{\"lam\":$LAMW,\"conservative\":1}"; done
for r in 1.0 0.5 0.2 0.05; do par --split test --policies ours --rho $r --budget $B --limit_q $NS --tag _cons --params "{\"lam\":$LAM,\"conservative\":1}"; done
par --split test --policies thompson --rho 0.1 --budget $B --limit_q $NQ --params "$TS"
W par --split test --policies thompson --rho 0.1 --budget $B --limit_q $NQW --tag _weak --params "$TS"
par --split test --policies llm_only --rho 0.1 --budget $B --limit_q $NL
for r in 1.0 0.5 0.2 0.05; do W par --split test --policies always --rho $r --budget $B --limit_q $NSW --tag _weak; done
for f in 0.5 2; do par --split test --policies ours --rho 0.1 --budget $B --limit_q $NS --tag _lamx$f --params "{\"lam\":$($PY -c "print($LAM*$f)")}"; done
for r in 1.0 0.5 0.2 0.05; do W par --split test --policies ours --rho $r --budget $B --limit_q $NSW --tag _weak --params "{\"lam\":$LAMW}"; done
$PY analyze.py > $L/analyze.log 2>&1
echo CORE_FINAL_DONE > $L/pipeline_final_core.done
for r in 1.0 0.5 0.2 0.05; do par --split test --policies topm --rho $r --budget $B --limit_q $NS --params "$TOPM"; done
for r in 1.0 0.5 0.2 0.05; do W par --split test --policies topm --rho $r --budget $B --limit_q $NSW --tag _weak --params "$TOPM"; done
$PY analyze.py > $L/analyze.log 2>&1
echo FINAL_DONE > $L/pipeline_final.done
