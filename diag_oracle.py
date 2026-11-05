#!/usr/bin/env python3
"""Ceiling test: train on REAL Botnet (80%) + benign, test on held-out REAL Botnet, per segment.
Shows how detectable each Botnet segment is in principle, with no GAN involved.
  python diag_oracle.py --data_dir ~/cicids/MachineLearningCVE --seg_cover 0.9
"""
import argparse, numpy as np
import paper_replica as pr
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir"); ap.add_argument("--seg_cover", type=float, default=None)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--trees", type=int, default=100)
ap.add_argument("--demo", action="store_true")
a = ap.parse_args()
if a.demo:
    Xdf, lab = pr.clean(pr.make_demo()); a.trees = 20
else:
    Xdf, lab = pr.load_clean(a.data_dir)
cols = list(Xdf.columns); y = lab.map(pr.to_group).values
Xr = Xdf.values.astype(np.float64); Xs = MinMaxScaler().fit(Xr).transform(Xr).astype(np.float32)
bot = np.where(y == "Botnet")[0]; Xb, Xbr = Xs[bot], Xr[bot]
segs = pr.segment_botnet(Xbr, cols, cover=a.seg_cover)
sid = np.zeros(len(bot), int)
for i, ix in enumerate(segs):
    sid[ix] = i
ben = np.where(y == "Benign")[0]; rng = np.random.default_rng(a.seed)
nb = min(10000, len(ben) // 2); pick = rng.permutation(ben)[:2 * nb]; bA, bB = pick[:nb], pick[nb:]
tr, te = train_test_split(np.arange(len(bot)), test_size=0.2, random_state=a.seed)
clf = RandomForestClassifier(n_estimators=a.trees, n_jobs=4, random_state=a.seed)
clf.fit(np.vstack([Xs[bA], Xb[tr]]), np.r_[np.zeros(nb), np.ones(len(tr))])
print(f"\nORACLE (trained on real Botnet): overall recall on held-out real Botnet {clf.predict(Xb[te]).mean():.2f}, "
      f"benign false-positive rate {clf.predict(Xs[bB]).mean():.4f}")
port = cols.index("Destination Port")
for i, ix in enumerate(segs):
    m = te[sid[te] == i]
    if len(m):
        print(f"  seg{i} n={len(ix):4d} port8080={(Xbr[ix][:, port] == 8080).mean():4.0%} "
              f"held-out={len(m):3d} recall={clf.predict(Xb[m]).mean():.2f}")
