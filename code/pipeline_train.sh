#!/bin/bash
# Encoder training (two rounds), calibration (two passes) and the LLM posterior model. Run from artifact/code.
set -e; PY=../venv/bin/python; L=../logs; mkdir -p $L
par() { for s in 0 1 2; do $PY run_policies.py "$@" --shard $s/3 > $L/$(echo "$@" | tr ' /' '__' | cut -c1-80)_$s.log 2>&1 & done; wait; }
if [ "$1" != "skip0" ]; then par --split train --collect behav --budget 30 --dummy; fi
$PY train_encoder.py behav --reset-calib > $L/train_encoder_r0.log 2>&1
par --split train --collect index --budget 30
$PY train_encoder.py behav,index --reset-calib > $L/train_encoder_r1.log 2>&1
par --split cal --collect index --budget 50 --tag _A --no_path
$PY fit_calib.py index_A > $L/fit_calib_A.log 2>&1
par --split cal --collect index --budget 50 --tag _B --no_path
$PY fit_calib.py index_B > $L/fit_calib_B.log 2>&1
par --split cal --collect index --budget 50 --tag _C --no_path
$PY fit_posterior.py index_C > $L/fit_posterior.log 2>&1
$PY fit_posterior.py index_C --post v0 --tag _v0 > $L/fit_posterior_v0.log 2>&1   # ablation: weighted multinomial fit
echo PIPELINE_DONE > $L/pipeline_train.done
