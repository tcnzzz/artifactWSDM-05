"""Strictly myopic scorer: a binary LightGBM head trained only on the immediate label (link target is a gold page),
on the same training rows as the multi-class encoder. Used by the `greedy_bin` policy."""
import os, glob, pickle, json, numpy as np, lightgbm as lgb
from common import D, FEATS
from crawl import MODELS
rows = []
for k in ("behav", "index"):
    for f in sorted(glob.glob(os.path.join(D, "rows", f"train_{k}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
qids = sorted(set(r["gid"].rsplit("_h", 1)[0] for r in rows)); rng = np.random.default_rng(0); held = set(rng.choice(qids, len(qids) // 10, replace=False))
tr = [r for r in rows if r["gid"].rsplit("_h", 1)[0] not in held]; va = [r for r in rows if r["gid"].rsplit("_h", 1)[0] in held]
Xt = np.vstack([r["X"] for r in tr]); yt = (np.concatenate([r["y"] for r in tr]) == 0).astype(int); Xv = np.vstack([r["X"] for r in va]); yv = (np.concatenate([r["y"] for r in va]) == 0).astype(int)
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, seed=0)
bst = lgb.train(params, lgb.Dataset(Xt, yt, feature_name=FEATS), 600, valid_sets=[lgb.Dataset(Xv, yv)], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])
bst.save_model(os.path.join(MODELS, "encoder_bin.txt"))
info = dict(n_train_links=int(len(yt)), positives=int(yt.sum()), best_iter=bst.best_iteration, heldout_logloss=float(bst.best_score["valid_0"]["binary_logloss"]))
json.dump(info, open(os.path.join(MODELS, "encoder_bin_info.json"), "w"), indent=1); print(info)
