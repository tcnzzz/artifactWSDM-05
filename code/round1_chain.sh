#!/bin/bash
# Wait for validation tuning, freeze the baselines, then run the confirmatory blocks.
cd "$(dirname "$0")"
while [ ! -f ../logs/round1_val.done ]; do sleep 60; done
../venv/bin/python round1_select.py > ../logs/round1_select.log 2>&1 || { echo SELECT_FAILED > ../logs/round1_chain.err; exit 1; }
NSH=12 ./round1_test.sh > ../logs/round1_test.log 2>&1
