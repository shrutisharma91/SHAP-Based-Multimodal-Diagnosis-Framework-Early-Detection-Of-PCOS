# PCOS SHAP multimodal

Polycystic ovary syndrome is often recognized late because the useful evidence is split across cycle history, hormones, and pelvic ultrasound. This project trains a tabular classifier and an ultrasound CNN, explains both with SHAP, and checks whether those explanations survive feature- and patch-dropping tests. Follicle measurements are part of the Rotterdam diagnosis, so the clinical model is fit again without them. The scores are decision support for research, not a diagnosis.

## Directory tree

```
pcos_shap_multimodal/
├── configs/config.yaml
├── data/
│   ├── raw/clinical/                 # Kaggle CSV pair or one xlsx
│   ├── raw/ultrasound/               # train|test / infected|notinfected
│   ├── interim/                      # ultrasound manifest
│   └── processed/                    # clinical train and test tables
├── src/
│   ├── data/                         # loaders, preprocessor, dedupe
│   ├── models/                       # XGBoost, EfficientNet-B0
│   ├── explain/                      # SHAP and faithfulness
│   ├── evaluation/
│   └── api/                          # FastAPI
├── scripts/                          # prepare, train, explain
├── app/streamlit_app.py
├── tests/
├── artifacts/                        # models, figures, reports
├── requirements.txt
└── Makefile
```

## How to run

```bash
make install
make prepare
# --- move repo to GPU machine, run these two there ---
make train-clinical
make train-image
# --- copy artifacts/ back, run these on your machine ---
make explain
make serve &  make app
```

`make prepare` builds the clinical tables only. After the ultrasound folders are in place, run `python -m scripts.prepare_images` so `make train-image` has `data/interim/ultrasound_manifest.csv`. On Windows, start `make serve` and `make app` in two terminals. The app reads `STREAMLIT_API_URL` (default `http://localhost:8000`).

## Data placement

Clinical files go in `data/raw/clinical/`. Use either one `.xlsx` whose sheets include `Model_Train_Scaled` and `Model_Test_Scaled`, or both Kaggle CSVs:

- `pcos_data_without_infertility.csv`
- `pcos_infertility.csv`

Leave only one of those sources in the folder. The CSVs are joined on `Patient File No.`.

Ultrasound JPEGs go under `data/raw/ultrasound/`:

```
data/raw/ultrasound/train/infected/
data/raw/ultrasound/train/notinfected/
data/raw/ultrasound/test/infected/
data/raw/ultrasound/test/notinfected/
```

`infected` is the positive class. Near-duplicate images are grouped by perceptual hash and a group that appears in both splits is kept on one side only.

## Paper contributions

- SHAP explanations for the clinical feature groups (symptoms, hormones, ultrasound, vitals, anthropometrics) and gradient SHAP overlays for ultrasound images.
- Faithfulness checks: top-k feature drop against a random drop, SHAP stability across folds and seeds, Rotterdam alignment of the top features, and an 8×8 patch-drop curve for images.
- Leakage-aware clinical feature sets: A keeps every feature, B drops ultrasound and follicle measures, C also drops pregnancy and beta-hCG, and D also drops AMH.
- A deduplication audit that clusters images with a 16×16 perceptual hash and removes cross-split near-duplicates before training.
- Recall-first thresholding: the highest score cutoff that still reaches recall 0.9, chosen on training out-of-fold or validation scores and then applied to the held-out test set.

## Limitations

- Clinical rows and ultrasound images are unpaired, so the fused score averages the two probabilities rather than training on matched patients.
- The published clinical workbook contains augmented rows. Those rows are flagged when they can be identified, and they are not an external cohort.
- There is no external validation set. Test metrics are a holdout from the same sources.
