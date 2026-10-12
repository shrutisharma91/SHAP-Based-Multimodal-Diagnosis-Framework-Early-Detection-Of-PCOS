# SHAP-based Multimodal Diagnosis Framework for Early Detection of PCOS

Two modalities, one explainable decision:

| Modality | Model | SHAP method | Output |
|---|---|---|---|
| Clinical / biochemical (hormones, BMI, cycle, symptoms, follicle counts) | XGBoost | TreeExplainer (exact) | Global feature ranking + per-patient waterfall |
| Ovarian ultrasound | EfficientNet-B0 / ResNet-50 (transfer learning) | GradientExplainer | Pixel attribution heatmaps |
| Fusion | Late fusion (stacking LR or weighted) | KernelExplainer | Clinical vs ultrasound contribution per patient |

Missing modality is handled: a patient with only labs or only a scan still gets a prediction.

## Structure

```
pcos_shap_multimodal/
├── configs/config.yaml            # all paths + hyperparameters
├── data/
│   ├── raw/clinical/              # PCOS_data_without_infertility.xlsx (or .csv)
│   ├── raw/ultrasound/{infected,notinfected}/
│   └── processed/paired_index.csv # patient_id,image_path,label  (for fusion)
├── src/
│   ├── utils.py
│   ├── data/        clinical.py · ultrasound.py
│   ├── models/      clinical_model.py · image_model.py · fusion.py
│   ├── explain/     shap_clinical.py · shap_image.py · shap_fusion.py
│   └── evaluation/  metrics.py
├── scripts/         train_clinical.py · train_image.py · train_fusion.py · predict.py
├── examples/sample_patient.json
├── tests/test_pipeline.py
└── artifacts/       (generated) models/ · figures/ · reports/
```

## Datasets

- Clinical: Kaggle "Polycystic ovary syndrome (PCOS)" — 541 patients, ~44 features (Kerala, India).
- Ultrasound: Kaggle "PCOS detection using ultrasound images" (infected / notinfected folders).

These two public datasets are **not from the same patients**. Train each unimodal model on its own dataset; the fusion layer needs paired data (`paired_index.csv`). If you have no paired data, report unimodal results + the fusion design, or simulate pairing by label for a proof-of-concept and state that clearly in the paper.

## Run

```bash
pip install -r requirements.txt
pytest -q                                   # smoke tests on synthetic data

python -m scripts.train_clinical
python -m scripts.train_image
python -m scripts.train_fusion              # needs paired_index.csv
python -m scripts.predict --clinical examples/sample_patient.json --image path/to/scan.jpg
```

Run all commands from the project root.

## Notes for the report

- Metrics are recall/sensitivity-focused (`best_threshold` picks the cutoff giving ≥90% sensitivity) — missing a case is costlier than a false alarm in screening.
- Ultrasound images in the Kaggle set include near-duplicate frames; if you can group by patient, split by patient to avoid leakage.
- Expected top SHAP drivers on the clinical side: follicle counts (L/R), AMH, cycle irregularity, weight gain, hair growth, skin darkening — consistent with Rotterdam criteria, which is a good validation point for the explainability chapter.
