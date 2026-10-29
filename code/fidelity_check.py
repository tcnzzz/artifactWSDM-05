"""Fidelity check between the two judge backends: the first cached calls (made through the Ollama daemon) are re-asked
through the llama.cpp server on the same model file, without touching the cache. Writes results/judge_backend_fidelity.json."""
import json, os, math, sqlite3, numpy as np, requests
from common import CACHE, R, Graph, SYS, llm_prompt, LLM_URL, load_goals
db = sqlite3.connect(os.path.join(CACHE, "llm.sqlite")); rows = db.execute("SELECT qid, parent, target, probs FROM insp WHERE model='qwen2.5:7b' ORDER BY rowid LIMIT 300").fetchall()
q = {}
for sp in ("cal", "train", "val", "test"):
    for g in load_goals(sp): q[g["qid"]] = g["question"]
G = Graph(); a = []; b = []
for qid, par, tgt, probs in rows:
    title, sents, links = G.page(par); an, si, _ = next(l for l in links if l[2] == tgt)
    j = requests.post(LLM_URL, json={"model": "qwen2.5:7b", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": llm_prompt(q[qid], title, sents[si], an)}],
                                      "max_tokens": 1, "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 20}, timeout=120).json()
    p = np.zeros(10)
    for t in j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        s = t["token"].strip()
        if len(s) == 1 and s in "0123456789": p[int(s)] += math.exp(t["logprob"])
    p = p / p.sum(); a.append(np.array(json.loads(probs))); b.append(p)
a = np.array(a); b = np.array(b); ea = a @ np.arange(10) / 9; eb = b @ np.arange(10) / 9
out = dict(n=len(a), argmax_agreement=float((a.argmax(1) == b.argmax(1)).mean()), mean_abs_diff_expected_rating=float(np.abs(ea - eb).mean()),
           pearson_expected_rating=float(np.corrcoef(ea, eb)[0, 1]), max_abs_diff_expected_rating=float(np.abs(ea - eb).max()))
json.dump(out, open(os.path.join(R, "judge_backend_fidelity.json"), "w"), indent=1); print(out)
