"""Train the cheap link encoder head (LightGBM on bi-encoder similarity features) on train goals only.
Target: layer of the link target (0 = gold page, 1 = one hop from gold, 2 = two hops, 3 = farther).
  python train_encoder.py behav            (round 0, rows from the behaviour policy)
  python train_encoder.py behav,index      (round 1, aggregated with rows from the round-0 index policy)
"""
import sys, os, glob, pickle, json
import numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from common import D, FEATS
from crawl import MODELS, K, Identity
if __name__ == "__main__":
    kinds = sys.argv[1].split(","); rows = []
    for k in kinds:
        for f in sorted(glob.glob(os.path.join(D, "rows", f"train_{k}_s*.pkl"))): rows += pickle.load(open(f, "rb"))
    qids = sorted(set(r["gid"].rsplit("_h", 1)[0] for r in rows)); rng = np.random.default_rng(0); held = set(rng.choice(qids, len(qids) // 10, replace=False))
    tr = [r for r in rows if r["gid"].rsplit("_h", 1)[0] not in held]; va = [r for r in rows if r["gid"].rsplit("_h", 1)[0] in held]
    Xt = np.vstack([r["X"] for r in tr]); yt = np.concatenate([r["y"] for r in tr]); Xv = np.vstack([r["X"] for r in va]); yv = np.concatenate([r["y"] for r in va])
    params = dict(objective="multiclass", num_class=K + 1, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=3, seed=0)
    bst = lgb.train(params, lgb.Dataset(Xt, yt, feature_name=FEATS), 600, valid_sets=[lgb.Dataset(Xv, yv)], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])
    bst.save_model(os.path.join(MODELS, "encoder.txt"))
    info = dict(kinds=kinds, n_train_links=int(len(yt)), n_heldout_links=int(len(yv)), class_counts=np.bincount(yt, minlength=K + 1).tolist(), best_iter=bst.best_iteration,
                heldout_logloss=float(bst.best_score["valid_0"]["multi_logloss"]), importance=dict(zip(FEATS, bst.feature_importance("gain").round(1).tolist())))
    json.dump(info, open(os.path.join(MODELS, "encoder_info.json"), "w"), indent=1); print(json.dumps(info, indent=1))
    if not os.path.exists(os.path.join(MODELS, "calib.pkl")) or "--reset-calib" in sys.argv:
        pickle.dump(dict(iso_prior=[Identity() for _ in range(K)], beta=1.0), open(os.path.join(MODELS, "calib.pkl"), "wb"))
