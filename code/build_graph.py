"""Parse the HotpotQA Wikipedia intro dump (2017-10-01) into a hyperlink graph with anchor text.

Output (data/graph/): pages.sqlite  (idx, title, body json {"s": sentences, "l": [[anchor, sent_idx, target_idx]]})
                      out_indptr.npy/out_idx.npy, in_indptr.npy/in_idx.npy (CSR, deduplicated), stats.json
Links whose target is not an article in the dump (redirects, red links) are dropped and counted.
"""
import tarfile, bz2, json, re, sqlite3, os, sys, html, time
from urllib.parse import unquote
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.environ.get("SF_DATA", os.path.join(HERE, "..", "data"))
SRC = os.environ.get("SF_SRC") or os.path.join(D, "raw", "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")
G = os.path.join(D, "graph"); os.makedirs(G, exist_ok=True)
A = re.compile(r'<a href="([^"]*)">(.*?)</a>', re.S); TAG = re.compile(r"<[^>]+>")

def parse_page(j):
    sents, links = [], []
    for para in ([j["text_with_links"]] if j["text_with_links"] and isinstance(j["text_with_links"][0], str) else j["text_with_links"]):
        for s in para:
            si = len(sents)
            for m in A.finditer(s):
                anchor = html.unescape(TAG.sub("", m.group(2))).strip()
                href = html.unescape(unquote(m.group(1))).split("#")[0].strip()
                if anchor and href: links.append([anchor, si, href])
            plain = html.unescape(TAG.sub("", s)).strip()
            sents.append(plain)
    return sents, links

def pass1():
    db = sqlite3.connect(os.path.join(G, "raw.sqlite")); db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
    db.execute("DROP TABLE IF EXISTS raw"); db.execute("CREATE TABLE raw(idx INTEGER PRIMARY KEY, title TEXT, body TEXT)")
    seen = set(); n = 0; t0 = time.time(); buf = []
    with tarfile.open(SRC, "r|bz2") as tf:
        for m in tf:
            if not m.isfile(): continue
            for line in bz2.decompress(tf.extractfile(m).read()).decode("utf-8").split("\n"):
                if not line.strip(): continue
                j = json.loads(line); title = html.unescape(j["title"])
                if title in seen: continue
                sents, links = parse_page(j)
                seen.add(title); buf.append((n, title, json.dumps({"s": sents, "l": links}, ensure_ascii=False))); n += 1
                if len(buf) >= 50000:
                    db.executemany("INSERT INTO raw VALUES(?,?,?)", buf); db.commit(); buf = []
                    print(n, round(time.time() - t0), flush=True)
    db.executemany("INSERT INTO raw VALUES(?,?,?)", buf); db.commit(); db.close()
    print("pass1 done", n, flush=True)

def pass2():
    """Resolve link targets to page indices; the intermediate and final tables live in separate files."""
    rd = sqlite3.connect(os.path.join(G, "raw.sqlite"))
    title2idx = {t: i for i, t in rd.execute("SELECT idx, title FROM raw")}; n = len(title2idx)
    def resolve(h):
        i = title2idx.get(h)
        if i is None and h: i = title2idx.get(h[0].upper() + h[1:])
        return -1 if i is None else i
    out_path = os.path.join(G, "pages.sqlite")
    if os.path.exists(out_path): os.remove(out_path)
    db = sqlite3.connect(out_path); db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE pages(idx INTEGER PRIMARY KEY, title TEXT, body TEXT)")
    indptr = np.zeros(n + 1, np.int64); tgt = []; n_raw = n_res = 0; buf = []
    for idx, title, body in rd.execute("SELECT idx,title,body FROM raw ORDER BY idx"):
        b = json.loads(body); seen = set(); out = []
        for anchor, si, href in b["l"]:
            n_raw += 1; t = resolve(href)
            if t < 0 or t == idx: continue
            n_res += 1
            if t in seen: continue              # keep the first link to each target
            seen.add(t); out.append([anchor, si, t])
        tgt.extend(x[2] for x in out); indptr[idx + 1] = len(tgt)
        buf.append((idx, title, json.dumps({"s": b["s"], "l": out}, ensure_ascii=False)))
        if len(buf) >= 50000:
            db.executemany("INSERT INTO pages VALUES(?,?,?)", buf); db.commit(); buf = []; print("pass2", idx, flush=True)
    db.executemany("INSERT INTO pages VALUES(?,?,?)", buf); db.commit(); rd.close()
    db.execute("CREATE INDEX t ON pages(title)"); db.commit(); db.close()
    tgt = np.asarray(tgt, np.int32)
    np.save(os.path.join(G, "out_indptr.npy"), indptr); np.save(os.path.join(G, "out_idx.npy"), tgt)
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr)); order = np.argsort(tgt, kind="stable")
    in_indptr = np.zeros(n + 1, np.int64); np.cumsum(np.bincount(tgt, minlength=n), out=in_indptr[1:])
    np.save(os.path.join(G, "in_indptr.npy"), in_indptr); np.save(os.path.join(G, "in_idx.npy"), src[order])
    deg = np.diff(indptr)
    stats = dict(pages=int(n), links_raw=int(n_raw), links_resolved=int(n_res), edges_dedup=int(len(tgt)),
                 out_degree_mean=float(deg.mean()), out_degree_median=float(np.median(deg)), out_degree_p90=float(np.percentile(deg, 90)),
                 pages_with_outlinks=int((deg > 0).sum()))
    json.dump(stats, open(os.path.join(G, "stats.json"), "w"), indent=1); print(stats)
    os.remove(os.path.join(G, "raw.sqlite"))
if __name__ == "__main__":
    if "--pass2-only" not in sys.argv: pass1()
    pass2()
