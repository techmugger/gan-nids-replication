#!/usr/bin/env python3
"""Charts from a finished run's CSV tables (ours vs paper). No retraining.
  python make_charts.py --run runC        -> runC/charts/*.png
"""
import argparse, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runC"); a = ap.parse_args()
T, O = f"{a.run}/tables", f"{a.run}/charts"; os.makedirs(O, exist_ok=True)
rd = lambda n: pd.read_csv(f"{T}/{n}.csv") if os.path.exists(f"{T}/{n}.csv") else None
C = {"Vanilla GAN": "tab:blue", "WGAN": "tab:orange", "CTGAN": "tab:green"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{O}/{name}.png", dpi=170); plt.close(fig); print("saved", f"{O}/{name}.png")

# 1. class distribution (Table 3)
t3 = rd("table3_regrouped")
if t3 is not None:
    t3 = t3.sort_values("instances", ascending=False)
    f, ax = plt.subplots(figsize=(8, 4.2)); b = ax.bar(t3["class"], t3["instances"], color="tab:blue")
    ax.set_yscale("log"); ax.set_ylabel("instances (log scale)"); ax.set_title("CIC-IDS2017 after regrouping (Table 3): severe class imbalance")
    for r, v in zip(b, t3["instances"]): ax.text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:,}", ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); save(f, "fig1_class_distribution")

# 2 & 6. per-class bars (Table 4 baseline, Table 8 WGAN x99)
def per_class(df, title, name):
    df = df[df["class"] != "ACCURACY"]; x = np.arange(len(df)); f, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (m, pm) in zip(axs, [("precision", "paper_P"), ("recall", "paper_R"), ("f1", "paper_F1")]):
        ax.bar(x - .2, df[m], .4, label="This work"); ax.bar(x + .2, df[pm], .4, label="Base paper")
        ax.set_xticks(x); ax.set_xticklabels(df["class"], rotation=40, ha="right"); ax.set_title(m.capitalize()); ax.set_ylim(0, 1.05)
    axs[0].legend(loc="lower left"); f.suptitle(title); save(f, name)
t4, t8 = rd("table4_baseline"), rd("table8_wgan99_all_classes")
if t4 is not None: per_class(t4, "Baseline NIDS (Random Forest), per class: this work vs base paper (Table 4)", "fig2_baseline_per_class")
if t8 is not None: per_class(t8, "WGAN x99 augmentation, per class (Table 8)", "fig6_wgan99_per_class")

# 3 & 4. metric vs multiplier (Tables 7 and 6)
def vs_mult(df, base, title, name):
    f, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, (m, pm, ttl) in zip(axs, [("P", "paper_P", "Precision"), ("R", "paper_R", "Recall"), ("F1", "paper_F1", "F1-score")]):
        for g, c in C.items():
            d = df[df["gan"] == g].sort_values("multiplier")
            ax.plot(d["multiplier"], d[m], "o-", color=c, label=f"{g} (this work)")
            ax.plot(d["multiplier"], d[pm], "s--", color=c, alpha=.6, label=f"{g} (paper)")
        if base is not None: ax.axhline(base[m], color="gray", ls=":", label="baseline (this work)")
        ax.set_xscale("log"); ax.set_xticks([4, 49, 99]); ax.set_xticklabels(["x4", "x49", "x99"]); ax.set_title(ttl); ax.set_ylim(0, 1.05)
    axs[0].legend(fontsize=7, loc="lower right"); f.suptitle(title); save(f, name)
base = None
if t4 is not None:
    r = t4[t4["class"] == "Botnet"].iloc[0]; base = {"P": r["precision"], "R": r["recall"], "F1": r["f1"]}
t7, t6 = rd("table7_original_botnet"), rd("table6_generated_test")
if t7 is not None: vs_mult(t7, base, "Botnet detection on ORIGINAL Botnet samples vs amount of GAN data (Table 7)", "fig3_table7_original_botnet")
if t6 is not None: vs_mult(t6, None, "Botnet detection on GENERATED Botnet samples (Table 6)", "fig4_table6_generated")

# 5. cosine heatmaps (Table 5)
t5 = rd("table5_cosine")
if t5 is not None:
    ours, pap = t5[~t5["model"].str.contains("paper")], t5[t5["model"].str.contains("paper")]
    f, axs = plt.subplots(1, 2, figsize=(14, 3.6), sharey=False)
    for ax, d, ttl in [(axs[0], ours, "This work"), (axs[1], pap, "Base paper")]:
        M = d.drop(columns="model").values.astype(float)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis"); ax.set_title(ttl)
        ax.set_yticks(range(len(d))); ax.set_yticklabels([m.replace(" (paper)", "") for m in d["model"]])
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(d.columns[1:], rotation=40, ha="right", fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    f.colorbar(im, ax=axs, fraction=.02); f.suptitle("Cosine similarity, generated vs original Botnet (Table 5)"); save(f, "fig5_cosine_similarity")
