# Artifact: Crawling as Pandora's Box with Two Inspection Costs

Budgeted focused crawling where every frontier link can be fetched blind (cost 1), inspected by
an LLM judge first (cost rho), or left alone, with links appearing only after their parent page
is fetched. Everything in the paper is produced by the scripts here from the saved runs and saved
result files in `results/`. Data provenance and checksums are in `DATA.md`, proofs and numeric
checks in `PROOFS.md`.

## Layout
- `code/download_dump.sh`, `code/build_graph.py`: Wikipedia intro-paragraph link graph (5.2M pages).
- `code/build_goals.py`: HotpotQA goals with seeds at exact depth 2, 3, 4; PageRank; BM25 statistics.
- `code/common.py`: graph access, cached bge-small embeddings, link features, LLM judge client.
- `code/crawl.py`: crawl simulator, chain index, all policies (`run`), model wrappers.
- `code/run_policies.py`: sharded, resumable runner (policies or training-row collection).
- `code/train_encoder.py`, `code/fit_calib.py`, `code/fit_posterior.py`: encoder head (train goals),
  isotonic calibration and beta (cal goals), LLM posterior model and inspection tables (cal goals).
- `code/estimate_lambda.py`: budget price from validation crawls.
- `code/eval_gain_curve.py`: held-out realised value of a judge call by encoder score (test of Proposition 1).
- `code/analyze.py`: all tables, figures and the `numbers.tex` macros of the manuscript.
- `code/check_prop1.py`, `code/check_prop2.py`: numeric checks of the propositions.
- `code/train_binary_head.py`: strictly myopic scorer (binary head) for the Proposition 2 test.
- `code/diag_index_vs_myopic.py`, `code/diag_redundancy.py`: diagnostics of the gateway null and of route redundancy.
- `code/gateway_synth.py`: exploratory synthetic-tree mechanism check for the gateway condition;
  writes `results/theory/gateway_synth.json` and is read only by the numbers step.
- `code/sensitivity.py`: exploratory sensitivity of the escalation region to lambda/rho and a
  budget-truncation boundary check; writes `results/exploratory/sensitivity.json`.
- `code/fidelity_check.py`: agreement of the two serving paths of the judge (same model file).
- Stages, in order: `code/pipeline_train.sh`, `code/stage_nollm.sh`, `code/pipeline_val.sh`,
  `code/pipeline_weak.sh` (zero-shot scorer regime), `code/pipeline_test.sh`, `code/pipeline_test_weak.sh`,
  `code/pipeline_final.sh` (conservative rule, sweeps, remaining baselines), then `code/diag_redundancy.py`,
  `code/fidelity_check.py` and `code/analyze.py`. `pipeline_extra.sh` is a
  superseded no-op.
- `results/runs/<split>/<policy>_rho<rho>/shard*.jsonl`: one record per crawl with the full event log
  (fetch or judge call, cumulative cost, gold flag, true layer, depth, index). Policy names: `ours_cons` is the
  paper's rule (conservative region), `ours` the point-estimate ablation, suffix `_weak` the zero-shot scorer.
- `results/theory/`, `results/tables/`, `results/gain_curve*.json`, `results/llm_usage.json`,
  `results/judge_backend_fidelity.json`.

## Reproduce
```
cd artifact && python3.13 -m venv venv && ./venv/bin/pip install numpy scipy scikit-learn lightgbm pandas pyarrow matplotlib torch sentence-transformers requests tqdm
cd code && ./download_dump.sh ../data/raw 1 && ../venv/bin/python build_graph.py && ../venv/bin/python build_goals.py
./pipeline_train.sh && ./stage_nollm.sh && ./pipeline_val.sh && ./pipeline_weak.sh && ./pipeline_test.sh && ./pipeline_final.sh
```
The LLM judge is `qwen2.5:7b` (Q4_K_M GGUF, digest in `DATA.md`), served either by Ollama
(`SF_LLM_URL=http://localhost:11434/api/chat`) or by a private llama.cpp server on the same model
file (default `http://localhost:11500/v1/chat/completions`). Judge outputs are cached in
`data/cache/llm.sqlite` keyed by (model, question, parent page, target page), so reruns are
deterministic and free. Seeds: goal construction 20261005, encoder 0, bootstrap 0, Thompson 0.

## Round-1 confirmatory analysis and the manuscript numbers
`code/round1_confirm.py` analyses the saved confirmatory runs (test questions 80-199 only) into
`results/tables/round1_confirm.json`; `code/inventory_runs.py` writes the per-cell crawl inventory
`results/tables/inventory.json`; `code/make_fig_confirm.py` draws Figure 1 from
`round1_confirm.json`; `code/gateway_synth.py` writes the exploratory synthetic check
`results/theory/gateway_synth.json`; `code/make_numbers_and_table.py` turns the round-0 numbers
(`manuscript/numbers_wsl_regenerated.tex`, the WSL output of `analyze.py`; the saved macOS
`numbers.tex` is kept for provenance) plus `round1_confirm.json` and `gateway_synth.json` into the
single active numbers file `manuscript/numbers_all.tex` and the table `manuscript/sections/tab_r1.tex`.
`MANIFEST.md` maps every macro used by the paper to its script and saved result. The paper builds with
`cd manuscript && latexmk -pdf main.tex`; no run file, log or frozen hash is written by these steps.
