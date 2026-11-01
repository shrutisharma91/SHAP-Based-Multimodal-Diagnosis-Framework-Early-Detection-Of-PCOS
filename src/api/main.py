"""FastAPI prediction service. Missing artifacts are reported as HTTP 503."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from src.api.schemas import ClinicalFeatures, HealthResponse, PredictionResponse
from src.data.clinical_preprocess import ClinicalPreprocessor, add_engineered_features
from src.data.ultrasound_loader import UltrasoundDataset, build_transforms
from src.explain.shap_clinical import ClinicalExplainer
from src.explain.shap_image import ImageExplainer
from src.models.clinical_model import load_model
from src.models.image_model import load_checkpoint
from src.utils import get_device, load_config, load_json

DISCLAIMER = "Research model only. This is not a diagnosis and does not replace clinical care."
LOGGER = logging.getLogger("src.api")
ROOT = Path(__file__).resolve().parents[2]
MODALITY = {
    "demographics": ("age_yrs", "weight_kg", "bmi", "waist_inch", "hip_inch"),
    "vitals": ("pulse_bpm", "hb_g_dl", "bp_systolic", "bp_diastolic"),
    "symptoms": ("cycle_ri", "cycle_length_days", "hair_growth", "pimples"),
    "hormones": ("fsh", "lh", "amh", "tsh", "prl", "beta_hcg_1", "beta_hcg_2"),
    "ultrasound": ("follicle_n_L", "follicle_n_R", "follicle_size_l_mm", "follicle_size_r_mm", "endometrium_mm"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts(app)
    yield


app = FastAPI(title="PCOS SHAP multimodal", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.clinical_model = None
app.state.preprocessor = None
app.state.clinical_explainer = None
app.state.image_model = None
app.state.image_explainer = None
app.state.img_size = 224
app.state.image_nsamples = 200
app.state.overlay_dir = ROOT / "artifacts" / "figures" / "overlays"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if app.state.clinical_model is not None or app.state.image_model is not None else "degraded",
        clinical_model=app.state.clinical_model is not None,
        image_model=app.state.image_model is not None,
    )


@app.post("/predict/clinical", response_model=PredictionResponse)
def predict_clinical(features: ClinicalFeatures) -> PredictionResponse:
    _require(app.state.clinical_model, "Clinical")
    probability = _clinical_probability(features)
    return _response(probability, _reasons(features), _modalities(features))


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    _require(app.state.image_model, "Image")
    probability, overlay = _image_probability(await _read_image(file))
    return _response(probability, [], ["image"], overlay)


@app.post("/predict/fused", response_model=PredictionResponse)
async def predict_fused(clinical: str = Form(...), file: UploadFile = File(...)) -> PredictionResponse:
    _require(app.state.clinical_model, "Clinical")
    _require(app.state.image_model, "Image")
    try:
        features = ClinicalFeatures.model_validate_json(clinical)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    clinical_probability = _clinical_probability(features)
    image_probability, overlay = _image_probability(await _read_image(file))
    modalities = _modalities(features)
    if "image" not in modalities:
        modalities.append("image")
    return _response((clinical_probability + image_probability) / 2.0, _reasons(features), modalities, overlay)


def _load_artifacts(app: FastAPI) -> None:
    config = load_config()
    artifacts = _resolve(config["paths"]["artifacts_dir"])
    processed = _resolve(config["paths"]["processed_dir"])
    app.state.img_size = int(config["image"]["img_size"])
    app.state.image_nsamples = int(config["shap"]["image_nsamples"])
    app.state.overlay_dir = _overlay_directory()
    app.state.device = get_device()
    _load_clinical(app, artifacts, processed, int(config["seed"]))
    _load_image(app, artifacts, config)


def _load_clinical(app: FastAPI, artifacts: Path, processed: Path, seed: int) -> None:
    model_path = artifacts / "models" / f"clinical_A_{_best_algo(artifacts)}.joblib"
    preprocessor_path = artifacts / "models" / "clinical_preprocessor.joblib"
    if not model_path.exists() or not preprocessor_path.exists():
        LOGGER.warning("Clinical artifacts missing. Looked for %s and %s", model_path, preprocessor_path)
        return
    try:
        app.state.preprocessor = ClinicalPreprocessor.load(preprocessor_path)
        app.state.clinical_model = load_model(model_path)
    except Exception:
        LOGGER.warning("Clinical model failed to load", exc_info=True)
        app.state.clinical_model = None
        app.state.preprocessor = None
        return
    train_path = processed / "clinical_train.parquet"
    if not train_path.exists():
        LOGGER.warning("Clinical SHAP background missing at %s", train_path)
        return
    try:
        frame, _ = _xy(pd.read_parquet(train_path))
        columns = app.state.preprocessor.feature_columns_
        background = frame.reindex(columns=columns).fillna(0.0)
        if background.empty:
            LOGGER.warning("Clinical SHAP background is empty")
            return
        background = background.sample(n=min(100, len(background)), random_state=seed)
        app.state.clinical_explainer = ClinicalExplainer(app.state.clinical_model, background)
    except Exception:
        LOGGER.warning("Clinical explainer failed to load", exc_info=True)
        app.state.clinical_explainer = None


def _load_image(app: FastAPI, artifacts: Path, config: dict) -> None:
    checkpoint = artifacts / "models" / "ultrasound_cnn.pt"
    if not checkpoint.exists():
        LOGGER.warning("Image checkpoint missing at %s", checkpoint)
        return
    try:
        app.state.image_model = load_checkpoint(checkpoint, app.state.device)
    except Exception:
        LOGGER.warning("Image model failed to load", exc_info=True)
        app.state.image_model = None
        return
    manifest_path = ROOT / "data" / "interim" / "ultrasound_manifest.csv"
    if not manifest_path.exists():
        LOGGER.warning("Image SHAP background manifest missing at %s", manifest_path)
        return
    try:
        manifest = pd.read_csv(manifest_path)
        train_rows = manifest.loc[manifest["split"] == "train"]
        if train_rows.empty:
            LOGGER.warning("Image SHAP background has no training rows")
            return
        sample_n = min(int(config["shap"]["image_background"]), len(train_rows))
        sample = train_rows.sample(n=sample_n, random_state=int(config["seed"]))
        dataset = UltrasoundDataset(sample, build_transforms(app.state.img_size, train=False))
        background = torch.stack([dataset[index][0] for index in range(len(dataset))])
        app.state.image_explainer = ImageExplainer(app.state.image_model, background, app.state.device)
    except Exception:
        LOGGER.warning("Image explainer failed to load", exc_info=True)
        app.state.image_explainer = None


def _clinical_probability(features: ClinicalFeatures) -> float:
    raw = pd.DataFrame([features.model_dump()])
    engineered = add_engineered_features(raw)
    aligned = engineered.reindex(columns=app.state.preprocessor.feature_columns_)
    transformed = app.state.preprocessor.transform(aligned).fillna(0.0)
    return _positive_proba(app.state.clinical_model, transformed)


def _reasons(features: ClinicalFeatures) -> list[dict]:
    explainer = app.state.clinical_explainer
    if explainer is None:
        return []
    raw = pd.DataFrame([features.model_dump()])
    aligned = add_engineered_features(raw).reindex(columns=explainer.feature_names)
    explanation = explainer.explain(app.state.preprocessor.transform(aligned).fillna(0.0))
    payload = explainer.patient_reasons(explanation, 0, top_k=5)
    reasons = []
    for direction, items in payload.items():
        reasons.extend({"direction": direction, **item} for item in items)
    return reasons


def _image_probability(image: Image.Image) -> tuple[float, str | None]:
    tensor = build_transforms(app.state.img_size, train=False)(image).unsqueeze(0)
    app.state.image_model.eval()
    with torch.no_grad():
        logit = app.state.image_model(tensor.to(app.state.device)).reshape(-1)
        probability = float(torch.sigmoid(logit).cpu())
    explainer = app.state.image_explainer
    if explainer is None:
        return probability, None
    stem = uuid.uuid4().hex
    values = explainer.explain(tensor, nsamples=app.state.image_nsamples)
    explainer.save_overlays(tensor, values, app.state.overlay_dir, [stem])
    return probability, f"/overlays/{stem}.png"


async def _read_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Image file is empty")
    try:
        return Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Upload is not a readable image") from exc


def _response(probability: float, reasons: list[dict], modalities: list[str], overlay: str | None = None) -> PredictionResponse:
    return PredictionResponse(
        p_pcos=probability,
        risk_band=_band(probability),
        top_reasons=reasons,
        modality_used=modalities,
        disclaimer=DISCLAIMER,
        overlay_path=overlay,
    )


def _modalities(features: ClinicalFeatures) -> list[str]:
    values = features.model_dump()
    return [name for name, fields in MODALITY.items() if any(values.get(field) is not None for field in fields)]


def _band(probability: float) -> str:
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Moderate"
    return "Low"


def _positive_proba(model, frame: pd.DataFrame) -> float:
    probabilities = model.predict_proba(frame)
    classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
    index = list(map(int, classes)).index(1) if 1 in set(map(int, classes)) else -1
    return float(probabilities[0, index])


def _best_algo(artifacts: Path) -> str:
    report = artifacts / "reports" / "clinical_results.json"
    if not report.exists():
        return "xgb"
    return load_json(report).get("sets", {}).get("A", {}).get("best_algo", "xgb")


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    drop = [column for column in ("PCOS", "is_synthetic_inferred", "patient_file_no", "sl_no") if column in frame.columns]
    return frame.drop(columns=drop), None


def _require(model, name: str) -> None:
    if model is None:
        raise HTTPException(status_code=503, detail=f"{name} model is not loaded. Train it and restart the server.")


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _overlay_directory() -> Path:
    try:
        directory = _resolve(load_config()["paths"]["artifacts_dir"]) / "figures" / "overlays"
    except Exception:
        directory = ROOT / "artifacts" / "figures" / "overlays"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


app.state.overlay_dir = _overlay_directory()
app.mount("/overlays", StaticFiles(directory=str(app.state.overlay_dir)), name="overlays")
