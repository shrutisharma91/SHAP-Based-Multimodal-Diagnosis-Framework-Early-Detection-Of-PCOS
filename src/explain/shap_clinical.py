"""SHAP explanations for the clinical XGBoost model (TreeExplainer: exact, fast)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import ensure_dir


class ClinicalExplainer:
    def __init__(self, model, background: pd.DataFrame | None = None) -> None:
        # Interventional SHAP w.r.t. a background set if given, else tree-path dependent
        self.explainer = shap.TreeExplainer(model, data=background, model_output="raw"
                                            if background is None else "probability")

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        return self.explainer(X)

    # ---------- global ----------
    @staticmethod
    def global_importance(sv: shap.Explanation, top_k: int = 15) -> pd.DataFrame:
        imp = np.abs(sv.values).mean(axis=0)
        df = pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": imp})
        return df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)

    @staticmethod
    def save_global_plots(sv: shap.Explanation, out_dir: str | Path, top_k: int = 15) -> None:
        out = ensure_dir(out_dir)

        shap.plots.beeswarm(sv, max_display=top_k, show=False)
        plt.title("Clinical features – SHAP summary")
        plt.tight_layout()
        plt.savefig(out / "clinical_shap_beeswarm.png", dpi=200, bbox_inches="tight")
        plt.close()

        shap.plots.bar(sv, max_display=top_k, show=False)
        plt.title("Clinical features – mean |SHAP|")
        plt.tight_layout()
        plt.savefig(out / "clinical_shap_bar.png", dpi=200, bbox_inches="tight")
        plt.close()

        # dependence plots for the top-3 drivers
        top = ClinicalExplainer.global_importance(sv, 3)["feature"]
        for feat in top:
            shap.plots.scatter(sv[:, feat], color=sv, show=False)
            plt.tight_layout()
            safe = "".join(ch if ch.isalnum() else "_" for ch in feat)
            plt.savefig(out / f"clinical_dependence_{safe}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # ---------- local (per patient) ----------
    @staticmethod
    def save_patient_waterfall(sv: shap.Explanation, idx: int, out_path: str | Path, top_k: int = 12) -> None:
        ensure_dir(Path(out_path).parent)
        shap.plots.waterfall(sv[idx], max_display=top_k, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    @staticmethod
    def patient_reasons(sv: shap.Explanation, idx: int, top_k: int = 5) -> dict[str, list[dict]]:
        """Human-readable drivers for one patient, split by direction."""
        row = pd.DataFrame({
            "feature": sv.feature_names,
            "value": sv.data[idx],
            "shap": sv.values[idx],
        })
        up = row[row.shap > 0].nlargest(top_k, "shap")
        down = row[row.shap < 0].nsmallest(top_k, "shap")
        to_list = lambda d: [  # noqa: E731
            {"feature": r.feature, "value": round(float(r.value), 3), "shap": round(float(r.shap), 4)}
            for r in d.itertuples()
        ]
        return {"increases_risk": to_list(up), "decreases_risk": to_list(down)}
