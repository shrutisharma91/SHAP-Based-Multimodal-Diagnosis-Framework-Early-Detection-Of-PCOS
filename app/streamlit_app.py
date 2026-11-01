"""Clinical and ultrasound tabs that call the prediction API."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

API_URL = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000").rstrip("/")
GROUPS = {
    "Symptoms": (
        ("cycle_ri", "Cycle regularity index"),
        ("cycle_length_days", "Cycle length (days)"),
        ("hair_growth", "Hair growth (0 or 1)"),
        ("pimples", "Pimples (0 or 1)"),
    ),
    "Hormones": (
        ("fsh", "FSH"),
        ("lh", "LH"),
        ("amh", "AMH"),
        ("tsh", "TSH"),
        ("prl", "Prolactin"),
        ("beta_hcg_1", "Beta-HCG 1"),
        ("beta_hcg_2", "Beta-HCG 2"),
    ),
    "Vitals": (
        ("pulse_bpm", "Pulse (bpm)"),
        ("hb_g_dl", "Hemoglobin (g/dL)"),
        ("bp_systolic", "Systolic BP"),
        ("bp_diastolic", "Diastolic BP"),
    ),
    "Anthropometric": (
        ("age_yrs", "Age (years)"),
        ("weight_kg", "Weight (kg)"),
        ("bmi", "BMI"),
        ("waist_inch", "Waist (inch)"),
        ("hip_inch", "Hip (inch)"),
    ),
    "Ultrasound measurements": (
        ("follicle_n_L", "Follicle count, left"),
        ("follicle_n_R", "Follicle count, right"),
        ("follicle_size_l_mm", "Follicle size, left (mm)"),
        ("follicle_size_r_mm", "Follicle size, right (mm)"),
        ("endometrium_mm", "Endometrium (mm)"),
    ),
}
BAND_COLOR = {"Low": "#1e8449", "Moderate": "#b9770e", "High": "#922b21"}


def main() -> None:
    st.set_page_config(page_title="PCOS SHAP", layout="wide")
    st.title("PCOS decision support")
    clinical, ultrasound = st.tabs(["Clinical", "Ultrasound"])
    with clinical:
        _clinical_tab()
    with ultrasound:
        _ultrasound_tab()
    st.divider()
    st.markdown(
        '<p style="color:red;font-weight:600;">Decision-support only; not a diagnosis. Confirm with Rotterdam criteria.</p>',
        unsafe_allow_html=True,
    )


def _clinical_tab() -> None:
    with st.form("clinical"):
        entered = {}
        for title, fields in GROUPS.items():
            st.subheader(title)
            left, right = st.columns(2)
            for index, (key, label) in enumerate(fields):
                column = left if index % 2 == 0 else right
                entered[key] = column.text_input(label, key=key, placeholder="optional")
        submitted = st.form_submit_button("Submit")
    if not submitted:
        return
    try:
        payload = _payload(entered)
        result = _post_json(f"{API_URL}/predict/clinical", payload)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return
    _risk(result)
    _waterfall(result.get("top_reasons") or [])
    _drivers(result.get("top_reasons") or [])


def _ultrasound_tab() -> None:
    upload = st.file_uploader("Ultrasound image", type=["jpg", "jpeg", "png"])
    if upload is None or not st.button("Submit", key="image_submit"):
        return
    try:
        result = _post_image(f"{API_URL}/predict/image", upload.name, upload.getvalue(), upload.type or "image/jpeg")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    st.metric("Probability of PCOS", f"{float(result['p_pcos']):.1%}")
    overlay = result.get("overlay_path")
    if overlay:
        st.image(f"{API_URL}{overlay}", caption="SHAP overlay")
    else:
        st.info("The API did not return a SHAP overlay.")


def _payload(entered: dict[str, str]) -> dict[str, float]:
    payload = {}
    for key, raw in entered.items():
        text = raw.strip()
        if not text:
            continue
        payload[key] = float(text)
    return payload


def _risk(result: dict) -> None:
    band = str(result.get("risk_band", ""))
    probability = min(max(float(result["p_pcos"]), 0.0), 1.0)
    color = BAND_COLOR.get(band, "#333333")
    st.markdown(
        f'<span style="background:{color};color:white;padding:6px 14px;border-radius:999px;font-weight:600;">{band}</span>',
        unsafe_allow_html=True,
    )
    st.progress(probability)
    st.caption(f"Probability of PCOS: {probability:.1%}")


def _waterfall(reasons: list[dict]) -> None:
    st.subheader("SHAP waterfall")
    if not reasons:
        st.info("The API did not return SHAP values for this patient.")
        return
    labels = [str(item.get("feature", "")) for item in reasons]
    values = [float(item.get("shap", 0.0)) for item in reasons]
    running = 0.0
    starts = []
    for value in values:
        starts.append(running)
        running += value
    figure, axis = plt.subplots(figsize=(8, 3.5))
    colors = ["#922b21" if value >= 0 else "#1a5276" for value in values]
    axis.bar(range(len(labels)), values, bottom=starts, color=colors, width=0.7)
    axis.axhline(0, color="#333333", linewidth=0.6)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Cumulative SHAP")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def _drivers(reasons: list[dict]) -> None:
    st.subheader("Top drivers")
    increasing = _driver_frame(reasons, "increases_risk")
    decreasing = _driver_frame(reasons, "decreases_risk")
    left, right = st.columns(2)
    left.markdown("**Increasing risk**")
    left.dataframe(increasing, hide_index=True, use_container_width=True)
    right.markdown("**Decreasing risk**")
    right.dataframe(decreasing, hide_index=True, use_container_width=True)


def _driver_frame(reasons: list[dict], direction: str) -> pd.DataFrame:
    rows = [
        {"feature": item.get("feature", ""), "shap": item.get("shap")}
        for item in reasons
        if item.get("direction") == direction
    ]
    return pd.DataFrame(rows, columns=["feature", "shap"])


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    return _read(request)


def _post_image(url: str, name: str, content: bytes, mime: str) -> dict:
    boundary = "----PcosBoundary7d9a"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = Request(
        url,
        data=head + content + tail,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _read(request)


def _read(request: Request) -> dict:
    try:
        with urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            detail = json.loads(raw).get("detail", raw)
        except json.JSONDecodeError:
            detail = raw
        if isinstance(detail, list):
            detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
        raise RuntimeError(str(detail)) from exc
    except URLError as exc:
        raise RuntimeError(f"API unreachable at {API_URL}: {exc.reason}") from exc


main()
