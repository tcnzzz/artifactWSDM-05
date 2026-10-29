"""Run crawl policies over a goal split. Resumable, sharded.
  python run_policies.py --split test --policies bfs,index --rho 0.1 --budget 50 --shard 0/3
  python run_policies.py --split train --collect behav --budget 30 --shard 0/3     (dump link rows for training)
"""
import argparse, json, os, sys, time, pickle, glob
import numpy as np
from common import Graph, Embedder, LLM, BM25, load_goals, D, G, R
from crawl import Crawl, Models, DummyModels, run, summarize, link_rows
LLM_POL = {"always", "topm", "thompson", "ours", "ours_online", "llm_only", "band"}
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test"); ap.add_argument("--policies", default=""); ap.add_argument("--collect", default="")
ap.add_argument("--rho", type=float, default=0.1); ap.add_argument("--budget", type=float, default=50); ap.add_argument("--shard", default="0/1")
ap.add_argument("--limit_q", type=int, default=0); ap.add_argument("--q_from", type=int, default=0); ap.add_argument("--q_to", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--params", default="{}")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dummy", action="store_true"); ap.add_argument("--no_path", action="store_true")
ap.add_argument("--llm_model", default="qwen2.5:7b"); ap.add_argument("--post_tag", default="")
a = ap.parse_args(); si, sn = map(int, a.shard.split("/")); P0 = json.loads(a.params); P0["seed"] = a.seed
goals = load_goals(a.split)
if a.limit_q: keep = list(dict.fromkeys(g["qid"] for g in goals))[:a.limit_q]; goals = [g for g in goals if g["qid"] in set(keep)]
if a.q_to: keep = list(dict.fromkeys(g["qid"] for g in goals))[a.q_from:a.q_to]; goals = [g for g in goals if g["qid"] in set(keep)]   # question range in file order
qids = sorted(set(g["qid"] for g in goals)); mine = set(q for i, q in enumerate(qids) if i % sn == si); goals = [g for g in goals if g["qid"] in mine]
graph = Graph(); emb = Embedder(); models = DummyModels() if a.dummy else Models(a.post_tag)
extras = dict(bm25=BM25(), pr=np.load(os.path.join(G, "pagerank.npy")))
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]; BUDGETS = [b for b in BUDGETS if b <= a.budget]
t0 = time.time()
if a.collect:
    out = os.path.join(D, "rows", f"{a.split}_{a.collect}{a.tag}_s{si}.pkl"); os.makedirs(os.path.dirname(out), exist_ok=True); rows = []
    for k, g in enumerate(goals):
        c = Crawl(g, graph, emb, models, a.rho, a.budget, extras=dict(extras, keepX=True)); run(a.collect, c, dict(P0))
        r = link_rows(c, force_path=not a.no_path); r["gid"] = g["gid"]; r["prior"] = None if a.dummy else c.Rp.copy()
        r["fetched"] = np.array(c.fetched); r["gold"] = np.array([g["p1"], g["p2"]]); rows.append(r)
        if k % 50 == 0: print(a.collect, k, len(goals), round(time.time() - t0), flush=True); emb.flush()
    pickle.dump(rows, open(out, "wb")); emb.flush(); sys.exit()
for pol in a.policies.split(","):
    P = dict(P0); name = f"{pol}{a.tag}_rho{a.rho:g}"; od = os.path.join(R, "runs", a.split, name); os.makedirs(od, exist_ok=True); of = os.path.join(od, f"shard{si}of{sn}.jsonl")
    done = set()                                       # crawls finished under any shard layout are skipped
    for f_ in glob.glob(os.path.join(od, "shard*.jsonl")):
        done |= set(json.loads(l)["gid"] for l in open(f_) if l.strip())
    llm = LLM(a.llm_model) if pol in LLM_POL else None
    if pol == "ours": P["tables"] = models.tables(a.rho, P["lam"], conservative=P.get("conservative", False))
    if pol == "band": P["tables"] = models.tables(a.rho, P["lam"])
    if pol == "ours_online": P["models"] = models
    with open(of, "a") as f:
        for k, g in enumerate(goals):
            if g["gid"] in done: continue
            c = Crawl(g, graph, emb, models, a.rho, a.budget, llm=llm, extras=extras); run(pol, c, P)
            f.write(json.dumps(summarize(c, BUDGETS)) + "\n"); f.flush()
            if k % 25 == 0:
                print(name, k, len(goals), round(time.time() - t0), "new_llm", llm.new_calls if llm else 0, flush=True); emb.flush()
                if llm: llm.flush()
    emb.flush()
    if llm:
        llm.flush(); json.dump(dict(new_calls=llm.new_calls, new_prompt_tokens=llm.new_ptok), open(os.path.join(od, f"llm_usage_shard{si}of{sn}.json"), "w"))
