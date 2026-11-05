#!/usr/bin/env python3
"""Per-segment diagnostic: which real Botnet segments does a benign-vs-synthetic classifier miss, and why?
Uses the cached GAN samples of an existing run (no retraining).
  python diag_segments.py --data_dir ~/cicids/MachineLearningCVE --run runA
"""
import argparse, numpy as np
import paper_replica as pr
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir"); ap.add_argument("--run", default="runA")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--trees", type=int, default=100)
ap.add_argument("--seg_cover", type=float, default=None); ap.add_argument("--demo", action="store_true")
a = ap.parse_args()
if a.demo:
    Xdf, lab = pr.clean(pr.make_demo()); a.trees = 20
else:
    Xdf, lab = pr.load_clean(a.data_dir)
cols = list(Xdf.columns); y = lab.map(pr.to_group).values
Xr = Xdf.values.astype(np.float64); Xs = MinMaxScaler().fit(Xr).transform(Xr).astype(np.float32)
bot = np.where(y == "Botnet")[0]; Xb, Xbr = Xs[bot], Xr[bot]
segs = pr.segment_botnet(Xbr, cols, cover=a.seg_cover); off = np.cumsum([0] + [len(s) for s in segs])
ben = np.where(y == "Benign")[0]; rng = np.random.default_rng(a.seed)
nb = min(10000, len(ben) // 2); bA = rng.permutation(ben)[:2 * nb][:nb]
port = cols.index("Destination Port")
for kind in pr.KINDS:
    S = np.load(f"{a.run}/cache/synth_{kind.replace(' ', '_')}.npz")["m1"]
    clf = RandomForestClassifier(n_estimators=a.trees, n_jobs=8, random_state=a.seed)
    clf.fit(np.vstack([Xs[bA], S]), np.r_[np.zeros(nb), np.ones(len(S))])
    print(f"\n{kind}")
    for i, ix in enumerate(segs):
        real, syn = Xb[ix], S[off[i]:off[i + 1]]
        lo, hi = syn.min(0), syn.max(0)
        inside = ((real >= lo - 1e-6) & (real <= hi + 1e-6)).mean(0)
        worst = np.argsort(inside)[:3]
        print(f"  seg{i} n={len(ix):4d} port8080={(Xbr[ix][:, port] == 8080).mean():4.0%} "
              f"recall={clf.predict(real).mean():.2f}  worst-covered: " +
              ", ".join(f"{cols[j]} {inside[j]:.0%}" for j in worst))
