#!/bin/bash
# Round 1 confirmatory runs on test questions 80..399 (never run with a judge before), in blocks of 40 questions in file
# order. Each block runs every cell (both regimes, rho in {0.1, 1}, scorer alone, TwoCost, band, top-m, Thompson over links).
# No new block starts after the deadline in ../preregistration/ROUND1_PLAN.md (Amendment 2). Resumable.
set -e; cd "$(dirname "$0")"; PY=${PY:-../venv/bin/python}; L=../logs; B=25; DEADLINE=${DEADLINE:-202610071600}
[ -f ../preregistration/round1_baselines.json ] || { echo "baselines not frozen"; exit 1; }
par() { N=${NSH:-12}; for s in $(seq 0 $((N-1))); do $PY run_policies.py "$@" --shard $s/$N > $L/r1t_$(echo "$@" | cksum | cut -d" " -f1)_$s.log 2>&1 & done; wait; }
cfg() { $PY -c "import json;c=json.load(open('../preregistration/round1_baselines.json'))['$1']['$2']['$3']['best'];print(json.dumps(c))"; }
for QF in 80 120 160 200 240 280 320 360; do
  QT=$((QF+40)); [ $(date +%Y%m%d%H%M) -lt $DEADLINE ] || { echo "deadline reached before block $QF" >> $L/round1_test.progress; break; }
  T="--split test --budget $B --q_from $QF --q_to $QT"
  for r in 0.1 1; do
    for REG in models models_weak; do
      export SF_MODELS=../data/$REG; S=$([ $REG = models_weak ] && echo _weak || echo ""); K=$([ $REG = models_weak ] && echo zeroshot || echo trained)
      LAM=$($PY -c "import json;print(json.load(open('../data/$REG/lambda.json'))['25']['lam'])")
      [ $r = 0.1 ] && par $T --policies index --rho 0.1 --tag _r1${S}
      par $T --policies ours --rho $r --tag _r1${S} --params "{\"lam\":$LAM,\"conservative\":\"all\"}"
      BC=$(cfg $K $r band); par $T --policies band --rho $r --tag _r1${S} --params "$($PY -c "import json;d=json.loads('$BC');d['lam']=$LAM;print(json.dumps(d))")"
      par $T --policies topm --rho $r --tag _r1${S} --params "$(cfg $K $r topm)"
      par $T --policies thompson --rho $r --tag _r1${S} --params "$(cfg $K $r thompson)"
    done
  done
  echo "block $QF-$QT done $(date '+%Y-%m-%d %H:%M')" >> $L/round1_test.progress
done
echo R1TEST_DONE > $L/round1_test.done
