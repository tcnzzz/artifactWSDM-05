"""Judge cost on this machine, from the llama.cpp server log of the run: per-call model time and overall throughput.
Timestamps in the log are minutes.seconds.milliseconds.microseconds since server start. Writes results/judge_latency.json."""
import re, json, os, sys, numpy as np
from common import R
HERE = os.path.dirname(os.path.abspath(__file__)); LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs", "llama_server.log")
pat = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I slot print_timing: .*total time =\s+([\d.]+) ms /\s+(\d+) tokens")
ts, tot, tok = [], [], []
for line in open(LOG, errors="ignore"):
    m = pat.match(line)
    if m:
        mi, se, ms, us, t, n = m.groups(); ts.append(int(mi) * 60 + int(se) + int(ms) / 1e3); tot.append(float(t)); tok.append(int(n))
ts = np.array(ts); tot = np.array(tot); tok = np.array(tok)
# throughput over busy minutes only (minutes with at least one completed call)
mins = np.floor(ts / 60); busy = np.unique(mins); per_min = np.array([(mins == b).sum() for b in busy])
out = dict(calls=int(len(tot)), median_call_ms=float(np.median(tot)), mean_prompt_tokens=float(tok.mean()), busy_minutes=int(len(busy)),
           calls_per_busy_minute_median=float(np.median(per_min)), seconds_per_call_at_median_throughput=float(60 / np.median(per_min)),
           device="Apple M5 Max (Metal), llama.cpp server from Ollama 0.30.8, 8 parallel slots, GPU shared with other users")
json.dump(out, open(os.path.join(R, "judge_latency.json"), "w"), indent=1); print(out)
