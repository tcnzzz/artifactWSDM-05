# Anonymous-supplement scrub report (2026-10-29)

Nothing was deleted in this session. This file lists what must be excluded or cleaned before an
anonymous submission, and the result of the identity scan. Paths are relative to the project root.

## Identity scan (what was actually found)

Commands: `grep -rIl` over `manuscript/`, `artifact/` (excluding `data/`, `.venv/`, `.venv-wsl/`) for
a macOS home-directory path, a Windows drive path (`/mnt/<x>/`, `D:\`, `C:\Users`), a machine hostname
(`LAPTOP-...`), e-mail addresses, and `Apple M*` / `MacBook` strings.

| location | finding | severity | action |
|---|---|---|---|
| `manuscript/**` (tex, bib, figures, old-draft) | none: no username, hostname, e-mail or absolute path in source or PDF metadata; `main.pdf` author field is empty (acmart anonymous mode) | clean | include as is |
| `artifact/code/**`, `artifact/preregistration/**` | none | clean | include |
| `artifact/results/**` | no path or username; `results/judge_latency.json` records the device as `Apple M5 Max (Metal), llama.cpp server from Ollama 0.30.8, 8 parallel slots, GPU shared with other users` | hardware fingerprint only | either keep (harmless for anonymity) or replace the `device` string with `laptop GPU, shared` |
| `artifact/logs/fit_posterior_r1.log` | 2 lines with an absolute macOS path under `/Users/<redacted>/Desktop/...` (lines 74, 76) | identifies the operator | exclude or scrub those lines |
| `artifact/logs/weak_fit_posterior_r1.log` | 2 lines with the same path (lines 22, 24) | identifies the operator | exclude or scrub |
| `artifact/logs/llama_server.log` | 1 line with a model path under `/Users/<redacted>/.ollama/...` (line 10); file is 4.2 GB | identifies the operator; huge | exclude |
| `manuscript/old-draft/main-old.pdf` | no author, but PDF metadata carries a `+07` timezone and the old `SemanticFrontier` title; the draft is superseded | stale + weak location hint | exclude |

No `/mnt/d/...`, `D:\...`, `C:\Users\...`, hostname, or e-mail string was found anywhere in the
project (including the 1,924 log files and 1,476 run files). The 4.2 GB server log is the only large
file in `artifact/logs/`; the other logs are small per-shard crawler logs.

## Files and directories to exclude from the anonymous supplement (do not delete)

| path | why | size |
|---|---|---|
| `manuscript/old-draft/` | superseded SemanticFrontier paper; not part of this submission | ~2 MB |
| `artifact/logs/llama_server.log` | 4.2 GB; one operator path | 4.2 GB |
| `artifact/logs/fit_posterior_r1.log`, `artifact/logs/weak_fit_posterior_r1.log` | operator path | ~4 KB |
| `artifact/data/raw/` | 1.55 GB Wikipedia dump; downloadable via the published SHA-256 | 1.55 GB |
| `artifact/data/cache/` (`llm.sqlite`, `emb.sqlite`) | caches of public data; large | 871 MB |
| `artifact/.venv/`, `artifact/.venv-wsl/` | local environments | — |
| `STATUS.md`, `PLAN.md`, `review.md`, `FROZEN_HASHES.md`, `artifact/SUPPLEMENT_SCRUB.md` | internal process/review files | small |
| `artifact/preregistration/ROUND1_PLAN.md` | internal preregistration; include only if the venue wants the preregistration, and note it contains process notes | small |
| `artifact/code/pipeline_*.sh`, `stage_nollm.sh`, `resume_all.sh`, `round1_test.sh` | local orchestration, harmless but unnecessary | small |

Safe to include: `manuscript/` (minus `old-draft/`), `artifact/code/*.py`, `artifact/results/runs/**`
(1,476 jsonl files), `artifact/results/tables/**`, `artifact/results/*.json`, `artifact/DATA.md`,
`artifact/README.md`, `artifact/MANIFEST.md`, `artifact/PROOFS.md`, the three preregistration files after
the author's decision.

## Leak check on the active paper

`manuscript/main.tex`, `manuscript/sections/*.tex`, `manuscript/references.bib` and `manuscript/main.pdf`
contain no name, e-mail, hostname, username or absolute path. Anonymity of the submission PDF does not
depend on the artifact scrub; the scrub matters only for a non-anonymous supplementary archive.
