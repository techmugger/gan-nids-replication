#!/usr/bin/env python3
"""
Faithful replication of: Zhao, Fok, Thing (2024) "Enhancing network intrusion detection
performance using generative adversarial networks", Computers & Security 145, 104005.

Produces (in --out):
  tables/  table1_counts, table3_regrouped, table4_baseline, table5_cosine,
           ml_validation (Sec 3.6.3), table6_generated_test, table7_original_botnet,
           table8_wgan99_all_classes, comparison_vs_paper
  figures/ fig2_vanilla_gan, fig3_wgan, fig4_ctgan   (cumulative sums, 8 features)
  run_config.json

Usage:
  python paper_replica.py --data_dir /path/to/MachineLearningCVE
  python paper_replica.py --demo          # tiny synthetic smoke test, NOT real results
"""
import argparse, glob, json, os, time, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
KINDS = ["Vanilla GAN", "WGAN", "CTGAN"]
MULTS = [4, 49, 99]
# the 8 features plotted in the paper's Figs 2-4 (text lists Fwd_Packets_s but figures show Flow_IAT_Mean)
FIG_FEATS = ["Flow Duration", "Total Length of Fwd Packets", "Flow Packets/s", "Flow IAT Mean",
             "Fwd IAT Mean", "Bwd IAT Mean", "Packet Length Mean", "Init_Win_bytes_backward"]

# ---------------------------------------------------------------- paper's published numbers
P_T1 = {"BENIGN": 2271320, "DoS Hulk": 230124, "PortScan": 158804, "DDoS": 128025, "DoS GoldenEye": 10293,
        "FTP-Patator": 7935, "SSH-Patator": 5897, "DoS slowloris": 5796, "DoS Slowhttptest": 5499, "Bot": 1956,
        "Web Attack Brute Force": 1507, "Web Attack XSS": 652, "Infiltration": 36,
        "Web Attack Sql Injection": 21, "Heartbleed": 11}
P_T4 = {"Benign": (1.00, 1.00, 1.00), "DoS": (0.98, 1.00, 0.99), "Probe": (0.99, 1.00, 1.00),
        "DDoS": (1.00, 1.00, 1.00), "Brute Force": (1.00, 1.00, 1.00), "Web Attack": (0.99, 0.97, 0.98),
        "Botnet": (0.87, 0.46, 0.60), "Infiltration": (1.00, 0.67, 0.80)}
P_T5 = {"Vanilla GAN": [.8979, .7608, .9673, .9013, .8962, .9669, .8750, .9999],
        "WGAN": [.8986, .7607, .9653, .8977, .8956, .9655, .8750, .9999],
        "CTGAN": [.8295, .5531, .9147, .8316, .8520, .8958, .7001, .9997]}
P_T6 = {("WGAN", 4): (.97, 1.00, .98), ("WGAN", 49): (1, 1, 1), ("WGAN", 99): (1, 1, 1),
        ("Vanilla GAN", 4): (.98, .99, .98), ("Vanilla GAN", 49): (1, 1, 1), ("Vanilla GAN", 99): (1, 1, 1),
        ("CTGAN", 4): (.99, .68, .81), ("CTGAN", 49): (.97, .92, .95), ("CTGAN", 99): (.95, .99, .97)}
P_T7 = {("WGAN", 4): (1, .74, .85), ("WGAN", 49): (1, .76, .87), ("WGAN", 99): (1, .82, .90),
        ("Vanilla GAN", 4): (1, .66, .80), ("Vanilla GAN", 49): (1, .76, .87), ("Vanilla GAN", 99): (1, .81, .90),
        ("CTGAN", 4): (1, .43, .60), ("CTGAN", 49): (1, .76, .86), ("CTGAN", 99): (1, .77, .87)}
P_T8 = {"Benign": (1, 1, 1), "DoS": (.98, 1, .99), "Probe": (.99, 1, 1), "DDoS": (1, 1, 1),
        "Brute Force": (1, 1, 1), "Web Attack": (.99, .98, .98), "Botnet": (1, 1, 1),
        "Infiltration": (1, .71, .83)}
P_MLVAL = {"CTGAN": (.98, .94, .96), "Vanilla GAN": (.99, .92, .95), "WGAN": (.99, .96, .98)}  # on testing set

GROUP = {"BENIGN": "Benign", "Bot": "Botnet", "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
         "DDoS": "DDoS", "DoS GoldenEye": "DoS", "DoS Hulk": "DoS", "DoS Slowhttptest": "DoS",
         "DoS slowloris": "DoS", "Heartbleed": "DoS", "PortScan": "Probe", "Infiltration": "Infiltration"}


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def norm_label(s):
    s = "".join(ch if ord(ch) < 128 else " " for ch in str(s))
    s = " ".join(s.split())
    return s.replace("Web Attack Sql", "Web Attack Sql").replace("Web Attack  ", "Web Attack ")


def to_group(l):
    return "Web Attack" if l.startswith("Web Attack") else GROUP[l]


# ---------------------------------------------------------------- data
def load_clean(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    assert files, f"no CSVs in {data_dir}"
    df = pd.concat([pd.read_csv(f, encoding="latin-1", low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return clean(df)


def clean(df):
    lab = df["Label"].map(norm_label)
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).values            # paper: drop rows with NaN/Null/Inf, nothing else
    return X[ok].reset_index(drop=True), lab[ok].reset_index(drop=True)


def make_demo():
    rng = np.random.default_rng(0)
    cols = ["Destination Port"] + FIG_FEATS + [f"f{i}" for i in range(78 - 9)]
    sizes = {"BENIGN": 6000, "DoS Hulk": 500, "PortScan": 400, "DDoS": 300, "DoS GoldenEye": 40, "FTP-Patator": 60,
             "SSH-Patator": 50, "DoS slowloris": 40, "DoS Slowhttptest": 40, "Bot": 300,
             "Web Attack Brute Force": 60, "Web Attack XSS": 30, "Infiltration": 36,
             "Web Attack Sql Injection": 21, "Heartbleed": 11}
    rows, labs = [], []
    for i, (k, n) in enumerate(sizes.items()):
        M = rng.lognormal(mean=i * 0.3, sigma=0.8, size=(n, len(cols)))
        M[:, 0] = rng.choice([80, 443, 8080], n) if k == "Bot" else rng.choice([80, 443, 22, 21], n)
        if k == "Bot":
            M[:, 10] = rng.integers(0, 2, n); M[:, 11] = rng.integers(0, 3, n)
        rows.append(M); labs += [k] * n
    df = pd.DataFrame(np.vstack(rows), columns=cols); df["Label"] = labs
    return df


# ---------------------------------------------------------------- Botnet segmentation (Sec 3.5)
def segment_botnet(Xr, cols, min_seg=30, max_cols=3, cover=None):
    """port==8080 vs rest, then split by columns that 'predominantly contain just two or three distinct values'.
    A column qualifies if it has 2-3 distinct values, or (when cover is set) its 3 most frequent values
    account for >= cover of the rows; other values are bucketed together."""
    port = Xr[:, cols.index("Destination Port")]
    segs = []
    for g in (np.where(port == 8080)[0], np.where(port != 8080)[0]):
        if len(g) == 0:
            continue
        sub = Xr[g]
        cand = []
        for j in range(sub.shape[1]):
            if cols[j] == "Destination Port":
                continue
            u, c = np.unique(sub[:, j], return_counts=True)
            top = np.argsort(c)[::-1][:3]
            if 2 <= len(u) <= 3 or (cover is not None and len(u) > 3 and c[top].sum() / c.sum() >= cover):
                keep = u[top]
                lab = np.where(np.isin(sub[:, j], keep), sub[:, j], -1e18)
                _, cc = np.unique(lab, return_counts=True); p = cc / cc.sum()
                cand.append((-(p * np.log(p)).sum(), j, keep))
        use = sorted(cand, key=lambda t: t[0], reverse=True)[:max_cols]
        if not use:
            segs.append(g); continue
        L = np.column_stack([np.where(np.isin(sub[:, j], keep), sub[:, j], -1e18) for _, j, keep in use])
        buckets = {}
        for i, k in enumerate(map(tuple, L)):
            buckets.setdefault(k, []).append(g[i])
        rest = []
        for b in buckets.values():
            (segs.append(np.array(b)) if len(b) >= min_seg else rest.extend(b))
        if rest:
            segs.append(np.array(rest))
    return segs


# ---------------------------------------------------------------- GANs (Sec 3.3-3.4)
def he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)


def make_nets(kind, n, noise):
    if kind in ("Vanilla GAN", "WGAN"):                       # Sec 3.4.1 / 3.4.2
        G = nn.Sequential(nn.Linear(noise, 25), nn.ReLU(), nn.Linear(25, 50), nn.ReLU(),
                          nn.Linear(50, n), nn.Sigmoid())
        D = [nn.Linear(n, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 1)]
        if kind == "Vanilla GAN":
            D.append(nn.Sigmoid())
        D = nn.Sequential(*D)
    else:                                                      # Sec 3.4.3 (conditional, dim/2dim/4dim)
        dg, dd = noise + 1, n + 1
        G = nn.Sequential(nn.Linear(dg, dg), nn.ReLU(), nn.Linear(dg, 2 * dg), nn.ReLU(),
                          nn.Linear(2 * dg, 4 * dg), nn.ReLU(), nn.Linear(4 * dg, n), nn.Sigmoid())
        D = nn.Sequential(nn.Linear(dd, 4 * dd), nn.ReLU(), nn.Dropout(0.1), nn.Linear(4 * dd, 2 * dd), nn.ReLU(),
                          nn.Dropout(0.1), nn.Linear(2 * dd, dd), nn.ReLU(), nn.Linear(dd, 1), nn.Sigmoid())
    G.apply(he); D.apply(he)
    return G, D


def train_gan(kind, X, steps, bs, noise, seed, dev, lr=None):
    seed_all(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev); n = X.shape[1]
    G, D = make_nets(kind, n, noise); G.to(dev); D.to(dev)
    cond = kind == "CTGAN"
    lab = lambda b: torch.ones(b, 1, device=dev)
    real = lambda b: Xt[torch.randint(0, len(Xt), (b,), device=dev)]
    gin = lambda b: torch.cat([torch.randn(b, noise, device=dev), lab(b)], 1) if cond else torch.randn(b, noise, device=dev)
    dw = lambda x: torch.cat([x, lab(len(x))], 1) if cond else x
    if kind == "WGAN":
        oG = torch.optim.RMSprop(G.parameters(), lr=lr or 5e-5); oD = torch.optim.RMSprop(D.parameters(), lr=lr or 5e-5)
        for _ in range(steps):
            for _ in range(5):
                x = real(bs); f = G(gin(bs)).detach()
                l = -(D(x).mean() - D(f).mean()); oD.zero_grad(); l.backward(); oD.step()
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)
            l = -D(G(gin(bs))).mean(); oG.zero_grad(); l.backward(); oG.step()
    else:
        bce = nn.BCELoss()
        oG = torch.optim.Adam(G.parameters(), lr=lr or 2e-4, betas=(0.5, 0.999))
        oD = torch.optim.Adam(D.parameters(), lr=lr or 2e-4, betas=(0.5, 0.999))
        one, zero = torch.ones(bs, 1, device=dev), torch.zeros(bs, 1, device=dev)
        for _ in range(steps):
            x = real(bs); f = G(gin(bs))
            ld = bce(D(dw(x)), one) + bce(D(dw(f.detach())), zero); oD.zero_grad(); ld.backward(); oD.step()
            lg = bce(D(dw(f)), one); oG.zero_grad(); lg.backward(); oG.step()
    G.eval()

    def sample(m):
        with torch.no_grad():
            return G(gin(m)).cpu().numpy().astype(np.float32)
    return sample


def snap(S, real, max_card=3):
    """Columns that take <=3 distinct values in the real segment (the paper's 'singular value' columns)
    are snapped to the nearest real value, so exact constants stay exact."""
    S = S.copy()
    for j in range(real.shape[1]):
        u = np.unique(real[:, j])
        if len(u) <= max_card:
            S[:, j] = u[np.abs(S[:, j][:, None] - u[None, :]).argmin(1)]
    return S


def build_synth(kind, Xb, segs, mults, a, dev, cache):
    path = os.path.join(cache, f"synth_{kind.replace(' ', '_')}.npz")
    if os.path.exists(path):                                   # resumable
        d = np.load(path); return {m: d[f"m{m}"] for m in mults}
    t = time.time()
    smp = [train_gan(kind, Xb[ix], a.steps, a.bs, a.noise, a.seed + i, dev, a.lr) for i, ix in enumerate(segs)]
    post = (lambda S, ix: snap(S, Xb[ix])) if a.snap else (lambda S, ix: S)
    out = {m: np.vstack([post(s(len(ix) * m), ix) for s, ix in zip(smp, segs)]) for m in mults}
    np.savez_compressed(path, **{f"m{m}": v for m, v in out.items()})
    print(f"  trained {kind} on {len(segs)} segments in {time.time() - t:.0f}s")
    return out


# ---------------------------------------------------------------- helpers
def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def prf_bin(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return round(P, 2), round(R, 2), round(F, 2)


def rf(a):
    return RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=a.seed)


def save(df, path):
    df.to_csv(path, index=False); print(f"  saved {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir"); ap.add_argument("--out", default="paper_replica_out")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--steps", type=int, default=3000, help="GAN training iterations per segment")
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--noise", type=int, default=25)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--lr", type=float, default=None, help="override GAN learning rate")
    ap.add_argument("--seg_cover", type=float, default=None,
                    help="also split on columns whose top-3 values cover >= this share (e.g. 0.9)")
    ap.add_argument("--snap_port", action="store_true",
                    help="snap generated Destination Port to a real port of the same segment (categorical column)")
    ap.add_argument("--snap", action="store_true", help="snap low-cardinality columns to real values")
    ap.add_argument("--validate_only", action="store_true", help="stop after Table 5/figs/ML validation")
    a = ap.parse_args()
    if a.demo:
        a.steps, a.trees, a.out = 150, 20, "paper_replica_demo"
    T, F, C = (os.path.join(a.out, d) for d in ("tables", "figures", "cache"))
    for d in (T, F, C):
        os.makedirs(d, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(a.seed)
    json.dump(vars(a) | {"device": dev, "torch": torch.__version__}, open(os.path.join(a.out, "run_config.json"), "w"), indent=2)

    # ---- 3.1 data, Table 1 / 3
    if a.demo:
        Xdf, lab = clean(make_demo())
    else:
        Xdf, lab = load_clean(a.data_dir)
    cols = list(Xdf.columns); print(f"{len(cols)} features, {len(Xdf)} rows after NaN/Inf removal")
    t1 = lab.value_counts().rename_axis("class").reset_index(name="ours")
    t1["paper"] = t1["class"].map(P_T1); t1["diff"] = t1["ours"] - t1["paper"]; save(t1, f"{T}/table1_counts.csv")
    y = lab.map(to_group).values
    t3 = pd.Series(y).value_counts().rename_axis("class").reset_index(name="instances"); save(t3, f"{T}/table3_regrouped.csv")
    Xr = Xdf.values.astype(np.float64)
    sc = MinMaxScaler().fit(Xr); Xs = sc.transform(Xr).astype(np.float32)

    # ---- 3.2 baseline (chi2 top-32, RF, 8:2)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=a.seed)
    sel = SelectKBest(chi2, k=32).fit(Xs[tr], y[tr]).get_support(indices=True)
    base = rf(a).fit(Xs[tr][:, sel], y[tr]); pb = base.predict(Xs[te][:, sel])
    acc = accuracy_score(y[te], pb); print(f"baseline accuracy {acc:.4f} (paper 0.9972)")
    rep = classification_report(y[te], pb, output_dict=True, zero_division=0)
    t4 = pd.DataFrame([{"class": k, "precision": round(rep[k]["precision"], 2), "recall": round(rep[k]["recall"], 2),
                        "f1": round(rep[k]["f1-score"], 2), "paper_P": P_T4[k][0], "paper_R": P_T4[k][1],
                        "paper_F1": P_T4[k][2]} for k in P_T4 if k in rep])
    t4.loc[len(t4)] = ["ACCURACY", acc, None, None, 0.9972, None, None]; save(t4, f"{T}/table4_baseline.csv")

    # ---- 3.5 botnet segmentation + GANs
    bot = np.where(y == "Botnet")[0]; Xb, Xbr = Xs[bot], Xr[bot]
    segs = segment_botnet(Xbr, cols, cover=a.seg_cover)
    print(f"botnet {len(bot)} samples -> {len(segs)} segments sizes {[len(s) for s in segs]}")
    synth = {k: build_synth(k, Xb, segs, [1] + MULTS, a, dev, C) for k in KINDS}

    if a.snap_port:
        pj = cols.index("Destination Port")
        for k in KINDS:
            for m, S in synth[k].items():
                S = S.copy(); o = 0
                for ix in segs:
                    n = len(ix) * m; u = np.unique(Xb[ix][:, pj])
                    S[o:o + n, pj] = u[np.abs(S[o:o + n, pj][:, None] - u[None, :]).argmin(1)]; o += n
                synth[k][m] = S

    # ---- 3.6.1 cosine similarity (Table 5): sorted real vs sorted generated, equal length
    fi = [cols.index(f) for f in FIG_FEATS]
    rows = []
    for k in KINDS:
        g = sc.inverse_transform(synth[k][1])
        rows.append([k] + [round(cosine(np.sort(Xbr[:, j]), np.sort(g[:, j])), 4) for j in fi])
    t5 = pd.DataFrame(rows, columns=["model"] + FIG_FEATS)
    for k in KINDS:
        t5.loc[len(t5)] = [k + " (paper)"] + P_T5[k]
    save(t5, f"{T}/table5_cosine.csv")

    # ---- 3.6.2 cumulative sums (Figs 2-4) on the largest segment
    big = int(np.argmax([len(s) for s in segs])); off = np.cumsum([0] + [len(s) for s in segs])
    for fig_name, k in zip(["fig2_vanilla_gan", "fig3_wgan", "fig4_ctgan"], KINDS):
        g = sc.inverse_transform(synth[k][1][off[big]:off[big + 1]]); r = Xbr[segs[big]]
        fg, ax = plt.subplots(4, 2, figsize=(11, 14))
        for a_, f, j in zip(ax.ravel(), FIG_FEATS, fi):
            a_.plot(np.cumsum(r[:, j]), label="Original"); a_.plot(np.cumsum(g[:, j]), label="Generated")
            a_.set_title(f"Comparison of Cumulative Sums for {f.replace(' ', '_')}", fontsize=8)
            a_.set_xlabel("Index"); a_.set_ylabel("Cumulative Sum"); a_.legend(fontsize=7)
        fg.suptitle(f"Cumulative sums for 8 features, {k} generated vs original Botnet samples")
        fg.tight_layout(); fg.savefig(f"{F}/{fig_name}.png", dpi=150); plt.close(fg)

    # ---- 3.6.3 ML validation (RF + DT) with 1956-equivalent synthetic + 10,000 benign
    ben = np.where(y == "Benign")[0]; rng = np.random.default_rng(a.seed)
    nb = min(10000, len(ben) // 2); pick = rng.permutation(ben)[:2 * nb]; bA, bB = pick[:nb], pick[nb:]
    ytest = np.r_[np.zeros(nb), np.ones(len(bot))]; Xtest = np.vstack([Xs[bB], Xb])
    rows = []
    for k in KINDS:
        Xd = np.vstack([Xs[bA], synth[k][1]]); yd = np.r_[np.zeros(nb), np.ones(len(synth[k][1]))]
        Xa, Xc, ya, yc = train_test_split(Xd, yd, test_size=0.2, random_state=a.seed, stratify=yd)
        for nm, clf in [("RF", rf(a)), ("DT", DecisionTreeClassifier(random_state=a.seed))]:
            clf.fit(Xa, ya)
            rows.append([k, nm, *prf_bin(yc, clf.predict(Xc)), *prf_bin(ytest, clf.predict(Xtest)),
                         *(P_MLVAL[k] if nm == "RF" else (None,) * 3)])
    save(pd.DataFrame(rows, columns=["gan", "clf", "P_20pct", "R_20pct", "F_20pct", "P_test", "R_test", "F_test",
                                     "paper_P_test", "paper_R_test", "paper_F_test"]), f"{T}/ml_validation.csv")

    if a.validate_only:
        print("\nvalidate_only: stopped after ML validation"); return

    # ---- Sec 4: augmentation (Tables 6, 7, 8): original Botnet REPLACED by synthetic
    nb_tr, nb_te = tr[y[tr] != "Botnet"], te[y[te] != "Botnet"]
    r6, r7, t8 = [], [], None
    for k in KINDS:
        for m in MULTS:
            S = synth[k][m][:, sel]; p = rng.permutation(len(S)); c = int(0.8 * len(S)); Str, Ste = S[p[:c]], S[p[c:]]
            Xtr = np.vstack([Xs[nb_tr][:, sel], Str]); ytr = np.r_[y[nb_tr], ["Botnet"] * len(Str)]
            clf = rf(a).fit(Xtr, ytr)
            Xg = np.vstack([Xs[nb_te][:, sel], Ste]); yg = np.r_[y[nb_te], ["Botnet"] * len(Ste)]
            pg = clf.predict(Xg)
            r6.append([k, m, *prf_bin(yg == "Botnet", pg == "Botnet"), *P_T6[(k, m)]])
            Xo = np.vstack([Xs[nb_te][:, sel], Xs[bot][:, sel]]); yo = np.r_[y[nb_te], ["Botnet"] * len(bot)]
            po = clf.predict(Xo)
            r7.append([k, m, *prf_bin(yo == "Botnet", po == "Botnet"), *P_T7[(k, m)]])
            print(f"  {k:12s} x{m:<3d} T6 {r6[-1][2:5]}  T7 {r7[-1][2:5]}")
            if k == "WGAN" and m == 99:
                rp = classification_report(yg, pg, output_dict=True, zero_division=0)
                t8 = pd.DataFrame([{"class": c_, "precision": round(rp[c_]["precision"], 2), "recall": round(rp[c_]["recall"], 2),
                                    "f1": round(rp[c_]["f1-score"], 2), "paper_P": P_T8[c_][0], "paper_R": P_T8[c_][1],
                                    "paper_F1": P_T8[c_][2]} for c_ in P_T8 if c_ in rp])
    hdr = ["gan", "multiplier", "P", "R", "F1", "paper_P", "paper_R", "paper_F1"]
    save(pd.DataFrame(r6, columns=hdr), f"{T}/table6_generated_test.csv")
    save(pd.DataFrame(r7, columns=hdr), f"{T}/table7_original_botnet.csv")
    save(t8, f"{T}/table8_wgan99_all_classes.csv")
    print("\nDone. Compare tables/*.csv columns ours vs paper_*; figures in", F)


if __name__ == "__main__":
    main()
