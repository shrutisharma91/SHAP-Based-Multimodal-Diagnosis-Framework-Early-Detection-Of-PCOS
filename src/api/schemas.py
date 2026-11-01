"""Pydantic v2 request and response models for the prediction API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class ClinicalFeatures(BaseModel):
    """Optional labs and ultrasound measures. At least five must be present."""

    age_yrs: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    pulse_bpm: Optional[float] = None
    hb_g_dl: Optional[float] = None
    cycle_ri: Optional[float] = None
    cycle_length_days: Optional[float] = None
    fsh: Optional[float] = None
    lh: Optional[float] = None
    amh: Optional[float] = None
    tsh: Optional[float] = None
    prl: Optional[float] = None
    beta_hcg_1: Optional[float] = None
    beta_hcg_2: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    waist_inch: Optional[float] = None
    hip_inch: Optional[float] = None
    follicle_n_L: Optional[float] = None
    follicle_n_R: Optional[float] = None
    follicle_size_l_mm: Optional[float] = None
    follicle_size_r_mm: Optional[float] = None
    endometrium_mm: Optional[float] = None
    hair_growth: Optional[float] = None
    pimples: Optional[float] = None

    @model_validator(mode="after")
    def require_five_values(self) -> "ClinicalFeatures":
        filled = sum(value is not None for value in self.model_dump().values())
        if filled < 5:
            raise ValueError("At least 5 non-null clinical features are required")
        return self


class PredictionResponse(BaseModel):
    p_pcos: float
    risk_band: Literal["Low", "Moderate", "High"]
    top_reasons: list[dict]
    modality_used: list[str]
    disclaimer: str
    overlay_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    clinical_model: bool = False
    image_model: bool = False
