"""Shared substrate: graph access, cached embeddings, link features, LLM inspection client."""
import os, json, sqlite3, re, math, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.environ.get("SF_DATA", os.path.join(HERE, "..", "data")); R = os.environ.get("SF_RESULTS", os.path.join(HERE, "..", "results"))
G = os.path.join(D, "graph"); CACHE = os.path.join(D, "cache"); os.makedirs(CACHE, exist_ok=True)
EMB_MODEL = "BAAI/bge-small-en-v1.5"; QPREFIX = "Represent this sentence for searching relevant passages: "
LLM_MODEL = "qwen2.5:7b"; OLLAMA = "http://localhost:11434/api/chat"
# A private llama.cpp server on the same GGUF blob as the Ollama model (OpenAI-compatible endpoint) avoids queueing behind
# other users of the shared Ollama daemon; the model, quantisation and prompt are identical. Set SF_LLM_URL to use it.
LLM_URL = os.environ.get("SF_LLM_URL", "http://localhost:11500/v1/chat/completions")
TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of in on at to for from by with and or is are was were be been which what who whom whose when where why how did does do that this these those it its as their his her he she they them has have had not also than then both same one two first other into about".split())
def toks(s): return [t for t in TOK.findall(s.lower()) if t not in STOP]

class Graph:
    def __init__(self):
        self.db = sqlite3.connect(f"file:{os.path.join(G, 'pages.sqlite')}?mode=ro", uri=True, check_same_thread=False)
        self.oi = np.load(os.path.join(G, "out_indptr.npy"), mmap_mode="r"); self.ox = np.load(os.path.join(G, "out_idx.npy"), mmap_mode="r")
        self.ii = np.load(os.path.join(G, "in_indptr.npy"), mmap_mode="r"); self.ix = np.load(os.path.join(G, "in_idx.npy"), mmap_mode="r")
        self.n = len(self.oi) - 1; self._c = {}
    def out(self, v): return np.asarray(self.ox[self.oi[v]:self.oi[v + 1]])
    def inn(self, v): return np.asarray(self.ix[self.ii[v]:self.ii[v + 1]])
    def page(self, v):
        """Fetch a page: (title, sentences, links=[(anchor, sent_idx, target)])."""
        v = int(v)
        if v not in self._c:
            if len(self._c) > 200000: self._c.clear()
            t, b = self.db.execute("SELECT title, body FROM pages WHERE idx=?", (v,)).fetchone(); b = json.loads(b)
            self._c[v] = (t, b["s"], b["l"])
        return self._c[v]
    def idx(self, title):
        r = self.db.execute("SELECT idx FROM pages WHERE title=?", (title,)).fetchone(); return None if r is None else r[0]
    def layers(self, gold, depth=2):
        """Backward BFS: int8 array with the hop distance from every page to the nearest gold page (capped at depth+1)."""
        dist = np.full(self.n, depth + 1, np.int8); cur = np.asarray(list(gold), np.int64); dist[cur] = 0
        for d in range(1, depth + 1):
            if len(cur) == 0: break
            nb = np.unique(np.concatenate([self.inn(v) for v in cur])) if len(cur) else np.array([], np.int64)
            nb = nb[dist[nb] > d]; dist[nb] = d; cur = nb
        return dist

class Embedder:
    """bge-small embeddings with an on-disk cache (pages keyed by index, free text keyed by hash)."""
    def __init__(self, device=None):
        self.model = None; self.device = device
        # autocommit: several processes write the cache concurrently, so no write transaction may stay open
        self.db = sqlite3.connect(os.path.join(CACHE, "emb.sqlite"), timeout=600, isolation_level=None); self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS page(idx INTEGER PRIMARY KEY, v BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS txt(h TEXT PRIMARY KEY, v BLOB)"); self.db.commit(); self._p = {}; self._n = 0
    def _enc(self, texts):
        if self.model is None:
            import torch; from sentence_transformers import SentenceTransformer
            dev = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = SentenceTransformer(EMB_MODEL, device=dev); self.model.max_seq_length = 128
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    def query(self, q):
        h = hashlib.sha1(("Q:" + q).encode()).hexdigest(); r = self.db.execute("SELECT v FROM txt WHERE h=?", (h,)).fetchone()
        if r: return np.frombuffer(r[0], np.float16).astype(np.float32)
        v = self._enc([QPREFIX + q])[0]; self.db.execute("INSERT OR REPLACE INTO txt VALUES(?,?)", (h, v.tobytes())); self.db.commit(); return v.astype(np.float32)
    def page(self, graph, idx):
        """Returns (title_vec, sent_vecs[S], anchor_vecs[L]) for a fetched page."""
        idx = int(idx)
        if idx in self._p: return self._p[idx]
        title, sents, links = graph.page(idx); S, L = len(sents), len(links)
        r = self.db.execute("SELECT v FROM page WHERE idx=?", (idx,)).fetchone()
        if r: m = np.frombuffer(r[0], np.float16).reshape(1 + S + L, -1)
        else:
            m = self._enc([title] + [s if s else title for s in sents] + [l[0] for l in links])
            self.db.execute("INSERT OR REPLACE INTO page VALUES(?,?)", (idx, m.tobytes())); self._n += 1
        m = m.astype(np.float32); out = (m[0], m[1:1 + S], m[1 + S:])
        if len(self._p) > 100000: self._p.clear()
        self._p[idx] = out; return out
    def flush(self): self.db.commit()

FEATS = ["cos_anchor", "cos_sent", "cos_ptitle", "cos_psent_max", "cos_psent_mean", "lex_q_in_anchor", "lex_anchor_in_q", "lex_q_in_sent",
         "sent_idx", "link_rank", "n_links", "parent_depth", "par_cos_anchor", "par_cos_sent", "cos_anchor_ptitle", "anchor_rank", "anchor_gap", "anchor_len"]
def link_features(graph, emb, qvec, qtok, page, depth, par_ca, par_cs):
    """Features of every link on a fetched page; only information visible before the link is followed."""
    title, sents, links = graph.page(page); L = len(links)
    if L == 0: return np.zeros((0, len(FEATS)), np.float32)
    tv, sv, av = emb.page(graph, page); sidx = np.array([l[1] for l in links])
    ca = av @ qvec; csent = sv @ qvec; cs = csent[sidx]; qs = set(qtok); X = np.zeros((L, len(FEATS)), np.float32)
    stok = [set(toks(s)) for s in sents]
    for i, (a, si, t) in enumerate(links):
        at = set(toks(a)); X[i, 5] = len(qs & at) / max(len(qs), 1); X[i, 6] = len(qs & at) / max(len(at), 1)
        X[i, 7] = len(qs & stok[si]) / max(len(qs), 1); X[i, 17] = len(at)
    X[:, 0] = ca; X[:, 1] = cs; X[:, 2] = tv @ qvec; X[:, 3] = csent.max(); X[:, 4] = csent.mean(); X[:, 8] = sidx
    X[:, 9] = np.arange(L) / L; X[:, 10] = L; X[:, 11] = depth; X[:, 12] = par_ca; X[:, 13] = par_cs; X[:, 14] = av @ tv
    X[:, 15] = (np.argsort(np.argsort(-ca))) / L; X[:, 16] = ca - ca.max()
    return X

class BM25:
    def __init__(self):
        j = json.load(open(os.path.join(G, "bm25_df.json"))); self.df = j["df"]; self.N = j["N"]; self.avgdl = j["avgdl"]
    def score(self, qtok, text, k1=1.2, b=0.75):
        d = toks(text); dl = len(d); s = 0.0
        if dl == 0: return 0.0
        for t in set(qtok):
            f = d.count(t)
            if f: s += math.log(1 + (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
        return s

SYS = "You judge hyperlinks for a web crawler that is collecting the pages needed to answer a question."
def llm_prompt(question, ptitle, sentence, anchor):
    return (f"Question: {question}\nCurrent page: {ptitle}\nSentence containing the link: {sentence}\nLink text: {anchor}\n\n"
            "How likely is it that the linked page is needed to answer the question, or leads towards such a page within one or two more links? "
            "Reply with a single digit from 0 (certainly not) to 9 (certainly yes).")
class LLM:
    """Pointwise link inspection through a local Ollama model; signal = digit distribution from token log-probabilities."""
    def __init__(self, model=LLM_MODEL):
        import requests; self.rq = requests.Session(); self.rq.headers.update({"Connection": "close"}); self.model = model
        # autocommit: several threads/processes write the cache concurrently, so no transaction may stay open across HTTP calls
        self.db = sqlite3.connect(os.path.join(CACHE, "llm.sqlite"), timeout=600, isolation_level=None, check_same_thread=False); self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS insp(model TEXT, qid TEXT, parent INTEGER, target INTEGER, probs TEXT, ptok INTEGER, PRIMARY KEY(model,qid,parent,target))"); self.db.commit()
        self.new_calls = 0; self.new_ptok = 0
    def inspect(self, qid, question, graph, parent, target):
        """Returns (expected rating in [0,1], P(rating>=5))."""
        r = self.db.execute("SELECT probs FROM insp WHERE model=? AND qid=? AND parent=? AND target=?", (self.model, qid, int(parent), int(target))).fetchone()
        if r: p = np.array(json.loads(r[0]))
        else:
            title, sents, links = graph.page(parent); a, si, _ = next(l for l in links if l[2] == target)
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": llm_prompt(question, title, sents[si], a)}]
            for attempt in range(6):
                try:
                    if "/v1/" in LLM_URL:
                        j = self.rq.post(LLM_URL, json={"model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 20}, timeout=(10, 120)).json()
                        lp = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]; ptok = j["usage"]["prompt_tokens"]
                    else:
                        j = self.rq.post(LLM_URL, json={"model": self.model, "messages": msgs, "stream": False, "logprobs": True, "top_logprobs": 20,
                                                        "options": {"temperature": 0, "num_predict": 1, "seed": 0, "num_ctx": 2048}, "keep_alive": "60m"}, timeout=300).json()
                        lp = j["logprobs"][0]["top_logprobs"]; ptok = j.get("prompt_eval_count", 0)
                    break
                except Exception as e:
                    time.sleep(2 + 5 * attempt)
                    if attempt == 5: raise
            p = np.zeros(10)
            for t in lp:
                s = t["token"].strip()
                if len(s) == 1 and s in "0123456789": p[int(s)] += math.exp(t["logprob"])
            p = p / p.sum() if p.sum() > 0 else np.full(10, 0.1)
            self.db.execute("INSERT OR REPLACE INTO insp VALUES(?,?,?,?,?,?)", (self.model, qid, int(parent), int(target), json.dumps([round(float(x), 5) for x in p]), ptok))
            self.new_calls += 1; self.new_ptok += ptok
        return float((p * np.arange(10)).sum() / 9.0), float(p[5:].sum())
    def flush(self): self.db.commit()

def load_goals(split):
    return [json.loads(l) for l in open(os.path.join(D, f"goals_{split}.jsonl"))]
