#!/bin/bash
# Weak-scorer regime (controlled factor): the cheap scorer is the zero-shot bge-small similarity of anchor and sentence,
# isotonically calibrated on cal goals; no trained head. Same judge, same posterior form, same tables. Run from artifact/code.
set -e; PY=../venv/bin/python; L=../logs; export SF_MODELS=../data/models_weak; mkdir -p $SF_MODELS
# single instance: a second caller waits for the first to finish
if [ -f $L/weak_train.lock ]; then while [ ! -f $L/weak_train.done ]; do sleep 30; done; exit 0; fi
touch $L/weak_train.lock; trap 'rm -f $L/weak_train.lock' EXIT
have() { ls ../data/rows/$1_s2.pkl > /dev/null 2>&1; }
par() { for s in 0 1 2; do $PY run_policies.py "$@" --shard $s/3 > $L/weak_$(echo "$@" | tr ' /' '__' | cut -c1-60)_$(echo "$@" | cksum | cut -c1-8)_$s.log 2>&1 & done; wait; }
have cal_dense_weak0 || { par --split cal --collect dense --budget 50 --tag _weak0 --no_path --dummy; $PY fit_calib.py dense_weak0 --dense > $L/weak_fit_calib0.log 2>&1; }
have cal_index_weakA || { par --split cal --collect index --budget 50 --tag _weakA --no_path; $PY fit_calib.py index_weakA --dense > $L/weak_fit_calib.log 2>&1; }
have cal_index_weakC || par --split cal --collect index --budget 50 --tag _weakC --no_path
cp ../data/models/lambda.json $SF_MODELS/lambda_strong.json
par --split val --policies index --rho 0.1 --budget 50 --tag _weak
SF_VALRUN=index_weak_rho0.1 $PY estimate_lambda.py > $L/weak_estimate_lambda.log 2>&1
$PY fit_posterior.py index_weakC --per_crawl 8 > $L/weak_fit_posterior.log 2>&1
echo WEAK_TRAIN_DONE > $L/weak_train.done
