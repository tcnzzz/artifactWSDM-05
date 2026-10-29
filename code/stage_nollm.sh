#!/bin/bash
set -e; PY=../venv/bin/python; L=../logs
par() { for s in 0 1 2; do $PY run_policies.py "$@" --shard $s/3 > $L/$(echo "$@" | tr ' /' '__' | cut -c1-70)_$s.log 2>&1 & done; wait; }
par --split val --policies index --rho 0.1 --budget 50
$PY estimate_lambda.py > $L/estimate_lambda.log 2>&1
par --split test --policies bfs,pagerank,bm25,dense,coverage,greedy,reach,index --rho 0.1 --budget 50
par --split test --collect index --budget 50 --tag _T --no_path
$PY train_binary_head.py > $L/train_binary_head.log 2>&1
par --split test --policies greedy_bin --rho 0.1 --budget 50
$PY diag_index_vs_myopic.py > $L/diag_index.log 2>&1
echo NOLLM_DONE > $L/stage_nollm.done
