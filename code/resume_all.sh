#!/bin/bash
# Resume everything after an interruption: finish the Thompson k1 validation cell, re-select hyperparameters, then the test
# pipeline (resumable: finished crawls are skipped).
cd "$(dirname "$0")"; PY=../venv/bin/python
for s in 0 1 2; do $PY run_policies.py --split val --policies thompson --rho 0.1 --budget 25 --limit_q 20 --tag _k1 --params '{"ts_kappa":1}' --shard $s/3 > ../logs/val_thompson_k1_$s.log 2>&1 & done; wait
$PY select_hparams.py > ../logs/select_hparams.log 2>&1
./pipeline_test.sh
