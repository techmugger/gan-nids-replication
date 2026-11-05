# GAN-based NIDS enhancement: a reproduction study

Reproduction of **Zhao, Fok & Thing (2024), "Enhancing network intrusion detection performance using generative adversarial networks", *Computers & Security* 145, 104005**, on the CIC-IDS2017 dataset.

The idea: attack classes such as Botnet are rare, so a detector misses them. GANs (Vanilla GAN, WGAN, CTGAN) generate synthetic Botnet flows, and the Random Forest NIDS is retrained on the augmented data.

![Working architecture](docs/architecture/working_architecture.png)

*Editable source: [`docs/architecture/working_architecture.drawio`](docs/architecture/working_architecture.drawio) (open at app.diagrams.net).*

## What reproduced and what did not

| Item | Base paper | This work |
|---|---|---|
| Class counts (15 classes, Tables 1 and 3) | given | identical (difference 0) |
| Baseline accuracy | 0.9972 | 0.9972 |
| Baseline Botnet precision / recall / F1 | 0.87 / 0.46 / 0.60 | 0.83 / 0.47 / 0.60 |
| Botnet F1, original flows, WGAN x49 (Table 7) | 0.87 | 0.80 |
| Botnet F1, original flows, WGAN x99 (Table 7) | 0.90 | 0.79 |
| Botnet F1, original flows, CTGAN x99 (Table 7) | 0.87 | 0.72 |
| Botnet F1, original flows, Vanilla GAN x99 (Table 7) | 0.90 | 0.62 |

The dataset, preprocessing and baseline reproduce exactly, and GAN augmentation raises Botnet F1 on original traffic from 0.60 to about 0.80 with WGAN, with diminishing returns beyond x49 as in the paper. **The paper's highest values (F1 up to 0.90) were not reached.** The paper does not state the GAN training length, learning rates, noise size, segmentation threshold, scaling, post-processing, Random Forest settings or random seed, so these were chosen by us (see `paper_replica.py`). The final settings were selected by comparing validation runs on real Botnet recall; all comparison runs are listed below.

| Run | Change | Recall on real Botnet (Vanilla GAN / WGAN / CTGAN) |
|---|---|---|
| 1 | 3,000 steps, strict segmentation, no snapping | 0.00 / 0.00 / 0.00 |
| 2 | 10,000 steps, snapping, 5 segments | 0.12 / 0.63 / 0.60 |
| 3 | 30,000 steps, snapping | 0.61 / 0.62 / 0.02 |
| 4 | 30,000 steps, no snapping | 0.00 / 0.00 / 0.00 |
| 5 (final) | 10,000 steps, snapping, 8 segments (`--seg_cover 0.9`) | 0.33 / 0.66 / 0.65 |
| 6 | run 5 plus port snapping (`--snap_port`) | 0.36 / 0.68 / 0.65 |

A Random Forest trained on real Botnet flows reaches recall 0.99 (`diag_oracle.py`), so the remaining gap lies in the generated samples of the non-8080 Botnet segments (`diag_segments.py`), not in the data.

Single runs, seed 42, no confidence intervals. Tables 6 and 8 of the paper test on generated data and are therefore a circular check; Table 7 (original Botnet flows) is the meaningful result.

## Repository contents

| File | Purpose |
|---|---|
| `paper_replica.py` | Full pipeline: cleaning, baseline, segmentation, GANs, similarity tests, augmentation (Tables 1 to 8, Figs 2 to 4) |
| `make_charts.py` | Charts comparing results with the paper |
| `confusion_matrices.py` | Confusion matrices of baseline and augmented models |
| `diag_segments.py` | Per-segment recall of classifiers trained on generated flows |
| `diag_oracle.py` | Ceiling test with a classifier trained on real Botnet flows |
| `paper_replica_notebook.ipynb` | Runs the pipeline in the background and shows tables and figures |
| `docs/` | Architecture diagram and result figures |

## Setup

Download CIC-IDS2017 (the `MachineLearningCSV.zip` / "MachineLearningCVE" folder with 8 CSV files) from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html). The dataset is not included here; please follow its terms and cite Sharafaldin et al. (2018).

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. quick check on synthetic data (about a minute)
python paper_replica.py --demo

# 2. final configuration used in this work (GAN training is the slow part; the generated samples are cached)
python -u paper_replica.py --data_dir /path/to/MachineLearningCVE --out runC \
       --steps 10000 --snap --seg_cover 0.9

# 3. charts, confusion matrices and diagnostics from a finished run
python make_charts.py --run runC
python confusion_matrices.py --data_dir /path/to/MachineLearningCVE --run runC
python diag_segments.py --data_dir /path/to/MachineLearningCVE --run runC --seg_cover 0.9
python diag_oracle.py   --data_dir /path/to/MachineLearningCVE --seg_cover 0.9
```

Useful options: `--validate_only` (stop after the similarity checks), `--snap_port`, `--trees`, `--seed`. Results are written to the `--out` folder (`tables/`, `figures/`, `cache/`). Copy the final `tables/*.csv` into `results/`.

## Reference

Zhao, X., Fok, K.W., Thing, V.L.L. (2024). Enhancing network intrusion detection performance using generative adversarial networks. *Computers & Security*, 145, 104005.

This repository is an independent reproduction for a university project; it contains no material from the paper.

## License

MIT, see `LICENSE`.
