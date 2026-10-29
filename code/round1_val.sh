#!/bin/bash
# Round-1 baseline tuning on validation questions 20..69 (150 crawls), per regime and per rho. Resumable.
set -e; cd "$(dirname "$0")"; PY=../venv/bin/python; L=../logs; B=25
par() { N=${NSH:-8}; for s in $(seq 0 $((N-1))); do $PY run_policies.py "$@" --shard $s/$N > $L/r1v_$(echo "$@" | cksum | cut -d" " -f1)_$s.log 2>&1 & done; wait; }
for REG in models models_weak; do
  export SF_MODELS=../data/$REG; S=$([ $REG = models_weak ] && echo _weak || echo "")
  LAM=$($PY -c "import json;print(json.load(open('../data/$REG/lambda.json'))['25']['lam'])")
  V="--split val --budget $B --q_from 20 --q_to 70"
  for r in 0.1 1.0; do
    par $V --policies ours --rho $r --tag _r1${S} --params "{\"lam\":$LAM,\"conservative\":\"all\"}"
    for m in 0 1 3 5; do for u in 0 1; do [ $m = 0 ] && [ $u = 0 ] && continue
      par $V --policies topm --rho $r --tag _r1${S}_m${m}u${u} --params "{\"m\":$m,\"mu\":$u}"; done; done
    for k in 3 10 30; do par $V --policies thompson --rho $r --tag _r1${S}_k$k --params "{\"ts_kappa\":$k}"; done
    for a in 0.25 0.5 1; do for b in 2 1000000000; do [ $(echo "$a < $b" | bc) = 1 ] || continue
      par $V --policies band --rho $r --tag _r1${S}_a${a}b${b} --params "{\"lam\":$LAM,\"a\":$a,\"b\":$b}"; done; done
  done
  par $V --policies index --rho 0.1 --tag _r1${S}
done
echo R1VAL_DONE > $L/round1_val.done
