# Data provenance

All data are public. Nothing is synthetic except the placement of the seed page, which is a
controlled factor and is labelled as such in the paper.

## 1. Wikipedia intro-paragraph hyperlink graph (HotpotQA processed dump)

- Source: https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
  (the processed English Wikipedia dump of 2017-10-01 released with HotpotQA; intro
  paragraphs with their hyperlinks and anchor text).
- Downloaded 2026-10-05 with `code/download_dump.sh` from the official host. How it was collected,
  stated plainly because it does not meet the polite-collection rule adopted afterwards: a first
  single-connection download ran at about 40 KB/s, so it was stopped and replaced by 32 parallel
  HTTP range requests to the same host from the same machine and IP (curl default User-Agent, curl
  `--retry 5`). No proxy, tunnel or change of IP was used, and no 429 response, block or
  Retry-After header was seen, but response codes were not logged, so this cannot be proven. The
  host was not rate limiting us that we know of; it was slow, and the parallel download was a
  convenience. It used one-off bandwidth of 1.55 GB. A re-run should use the script with `N=1`
  (one connection) and a descriptive User-Agent; the checksum below lets anyone verify they got
  the same file. Size 1,553,565,403 bytes. SHA-256:
  `1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729`
  (also in `data/raw/SHA256SUMS`).
- Licence: Wikipedia text, CC BY-SA 4.0 (and GFDL). The HotpotQA release is CC BY-SA 4.0.
- Processing (`code/build_graph.py`): every article becomes a page with its intro
  sentences; every `<a href>` in the intro becomes a link with its anchor text, sentence
  index and target article. Targets that are not articles of the dump (redirects, red
  links) are dropped; links are deduplicated per (page, target), keeping the first.
  Statistics are in `data/graph/stats.json`: 5,233,329 pages, 23,392,247 raw links,
  19,467,768 resolved, 19,262,362 distinct edges, mean out-degree 3.68 (median 3, p90 7).
- The crawler only ever sees fetched pages and the anchor text and sentence of their
  links. PageRank on the full graph (`data/graph/pagerank.npy`) is computed for the
  PageRank baseline only and is marked as oracle graph knowledge in the paper.

## 2. HotpotQA questions (goals)

- Collection: read from a Hugging Face cache that already existed on this machine; no new
  download was made for this project.

- Source: Hugging Face dataset `hotpotqa/hotpot_qa`, config `distractor`, snapshot
  `1908d6afbbead072334abe2965f91bd2709910ab` (local cache). Parquet SHA-256:
  - train-00000-of-00002.parquet `76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68`
  - train-00001-of-00002.parquet `713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f`
  - validation-00000-of-00001.parquet `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`
- Licence: CC BY-SA 4.0 (Yang et al., EMNLP 2018).
- Goal construction (`code/build_goals.py`, seed 20261005): bridge questions whose two
  gold supporting articles both exist in the graph and are linked in exactly one direction
  P1 -> P2. For each question and each h in {1, 2, 3} a seed page is drawn by a backward
  random walk of h steps from P1 (uniform over in-neighbours, seed out-degree in [3, 80]),
  kept only if the exact shortest-path distances are dist(seed, P1) = h and
  dist(seed, P2) = h + 1. Target depth is therefore known exactly (2, 3 or 4 hops to P2).
  A question is used only if all three depths could be constructed.
- Splits (`data/goals_stats.json`): test 400 questions (1,200 goals) from the HotpotQA
  dev set; val 150, cal 400, train 771 questions from the HotpotQA train set (a 12,000
  question sample, 8,714 candidates, exhausted after the gold-page disjointness filter).
  No gold page of any goal appears as a gold page in another split. All model fitting uses
  train (encoder head), cal (calibrators, beta, LLM posterior model, inspection tables) and
  val (budget price, baseline hyperparameters); test is touched only for the reported runs.

## 3. Models

- Collection: bge-small-en-v1.5 was read from an existing local Hugging Face cache and the judge
  from an existing local Ollama model store. No web page was crawled live in any experiment: all
  crawls run against the static local graph.

- Link encoder: `BAAI/bge-small-en-v1.5` (snapshot `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`,
  MIT licence) sentence embeddings plus a LightGBM 4.7.0 multiclass head trained in
  `code/train_encoder.py`.
- LLM judge: `qwen2.5:7b` (Ollama model blob
  `sha256:2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730`, GGUF Q4_K_M). The first
  300 cached calls went through the Ollama daemon; every other call went through a private llama.cpp
  server (the `llama-server` binary bundled with Ollama 0.30.8) loading the same blob:
  `llama-server -m <blob> --port 11500 -ngl 99 -c 8192 -np 4 -t 4 --threads-http 16`, OpenAI chat
  endpoint, temperature 0, `max_tokens` 1, top-20 log-probabilities. All experiments, calibration
  and test, use this one laptop backend (Apple Silicon, Metal). Judge outputs are cached by
  (model, question, parent, target), so a link is judged once and every policy sees the same signal.
  Fidelity check (`code/fidelity_check.py`, `results/judge_backend_fidelity.json`): the 300 Ollama-era
  prompts re-asked through llama.cpp give the same top digit in 300 of 300 cases, a mean absolute
  difference of 0.0003 in the expected rating (on a 0 to 1 scale, maximum 0.005) and a Pearson
  correlation of 0.99999. The cached Ollama values were kept. Cross-model replication (if present in
  `results/`): `llama3.1:8b` under the same prompt.
- Software: Python 3.13, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0,
  torch 2.14.1, sentence-transformers 6.1.0, transformers 5.18.0.
- Reproducibility caveat (beta): `data/models/calib.pkl` stores beta = 2.5846579129232894.
  Refitting `code/fit_calib.py index_B` on the saved `data/rows/cal_index_B_s*.pkl` (byte-identical
  to the 2026-10-29 backup) gives 2.5473 with the default `numpy.argsort` and 2.5660 with a stable
  sort; n_links (310,397) and n_pages_for_beta (1,447) match the stored fit, and the isotonic curves
  agree at the fitted quantiles. The gap is tie-breaking in the rank statistic, which is
  platform-dependent, not row drift. All saved runs and paper numbers use the frozen beta;
  off-platform regeneration will give a slightly different beta unless the original sort order is
  reproduced. Diagnostic: `code/level_shift.py` -> `results/exploratory/level_shift.json`.
