#!/usr/bin/env python3
"""Confusion matrices from a finished run. Retrains only the Random Forests (~20 min), reuses cached GAN samples.
  python confusion_matrices.py --data_dir ~/cicids/MachineLearningCVE --run runC
Outputs in <run>/confusion/: cm_baseline.png, cm_<gan>_x99_8class.png (Table 8 style), cm_table7_binary_x99.png
"""
import argparse, os, numpy as np
import paper_replica as pr
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CL = ["Benign", "DoS", "Probe", "DDoS", "Brute Force", "Web Attack", "Botnet", "Infiltration"]
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir"); ap.add_argument("--run", default="runC")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--trees", type=int, default=100)
ap.add_argument("--demo", action="store_true")
a = ap.parse_args()
if a.demo:
    Xdf, lab = pr.clean(pr.make_demo()); a.trees = 20
else:
    Xdf, lab = pr.load_clean(a.data_dir)
O = f"{a.run}/confusion"; os.makedirs(O, exist_ok=True)
y = lab.map(pr.to_group).values; Xr = Xdf.values.astype(np.float64)
Xs = MinMaxScaler().fit(Xr).transform(Xr).astype(np.float32)
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=a.seed)
sel = SelectKBest(chi2, k=32).fit(Xs[tr], y[tr]).get_support(indices=True)
rf = lambda: RandomForestClassifier(n_estimators=a.trees, n_jobs=8, random_state=a.seed)

def draw(ax, yt, yp, labels, title):
    cm = confusion_matrix(yt, yp, labels=labels); rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, vmin=0, vmax=1, cmap="Blues"); ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]: ax.text(j, i, f"{cm[i, j]:,}\n{100*rn[i, j]:.2g}%", ha="center", va="center",
                                 color="w" if rn[i, j] > .5 else "k", fontsize=7 if len(labels) > 4 else 11)

def save(f, name):
    f.tight_layout(); f.savefig(f"{O}/{name}.png", dpi=170); plt.close(f); print("saved", f"{O}/{name}.png")

base = rf().fit(Xs[tr][:, sel], y[tr]); pb = base.predict(Xs[te][:, sel])
f, ax = plt.subplots(figsize=(8, 7)); draw(ax, y[te], pb, CL, "Baseline NIDS (RF), 8-class confusion matrix\n(cell text: count and share of the true class;\ncolour: share of the true class)"); save(f, "cm_baseline")

bot = np.where(y == "Botnet")[0]; nb_tr, nb_te = tr[y[tr] != "Botnet"], te[y[te] != "Botnet"]
rng = np.random.default_rng(a.seed)
Xo = np.vstack([Xs[nb_te][:, sel], Xs[bot][:, sel]]); yo = np.r_[y[nb_te], ["Botnet"] * len(bot)]
f2, axs = plt.subplots(1, 4, figsize=(17, 4)); bin_ = lambda v: np.where(v == "Botnet", "Botnet", "Not Botnet")
draw(axs[0], bin_(y[te]), bin_(pb), ["Botnet", "Not Botnet"], "Baseline (real data)")
for k, ax in zip(pr.KINDS, axs[1:]):
    S = np.load(f"{a.run}/cache/synth_{k.replace(' ', '_')}.npz")["m99"][:, sel]
    p = rng.permutation(len(S)); c = int(.8 * len(S))
    clf = rf().fit(np.vstack([Xs[nb_tr][:, sel], S[p[:c]]]), np.r_[y[nb_tr], ["Botnet"] * c])
    Xg = np.vstack([Xs[nb_te][:, sel], S[p[c:]]]); yg = np.r_[y[nb_te], ["Botnet"] * (len(S) - c)]
    f, a8 = plt.subplots(figsize=(8, 7)); draw(a8, yg, clf.predict(Xg), CL, f"{k} x99: 8-class confusion (generated Botnet in test, Table 8 setting)")
    save(f, f"cm_{k.replace(' ', '_').lower()}_x99_8class")
    draw(ax, bin_(yo), bin_(clf.predict(Xo)), ["Botnet", "Not Botnet"], f"{k} x99 (original Botnet test)")
f2.suptitle("Table 7 setting: Botnet vs rest, tested on ORIGINAL Botnet samples"); save(f2, "cm_table7_binary_x99")
