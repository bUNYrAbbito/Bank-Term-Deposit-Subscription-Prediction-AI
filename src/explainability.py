"""
Explainability module for Bank Term Deposit Prediction.
Provides model interpretability via Gini / Gain Feature Importances and
SHAP (SHapley Additive exPlanations) values, beeswarm plots, and local waterfall explanations.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def compute_feature_importances(
    model: Any, feature_names: List[str]
) -> pd.DataFrame:
    """
    Extracts native feature importances from tree-based estimators.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute.")

    df_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df_imp


def plot_top_feature_importances(
    df_imp: pd.DataFrame, model_name: str, save_path: Path, top_n: int = 20
):
    """
    Plots a horizontal bar chart of the top N most important features.
    """
    df_top = df_imp.head(top_n).sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(df_top["Feature"], df_top["Importance"], color="#1f77b4")
    plt.title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    save_dir: Path,
    n_background: int = 300,
) -> Tuple[shap.Explanation, shap.TreeExplainer]:
    """
    Computes SHAP values using TreeExplainer and exports summary plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use TreeExplainer for tree ensembles
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Convert to standard format if multi-class or binary
    if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
        # Binary classification class 1 SHAP values
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # 1. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_class1.values,
        features=X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Feature Attribution (Impact on P(Subscription = Yes))", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. SHAP Bar Plot (Global Importance)
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values_class1, max_display=20, show=False)
    plt.title("Mean Absolute SHAP Value (Global Impact)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_feature_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Local Waterfall Plot for a High Probability instance
    probas = model.predict_proba(X_sample)[:, 1]
    high_idx = int(np.argmax(probas))
    low_idx = int(np.argmin(probas))

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_class1[high_idx], max_display=12, show=False)
    plt.title(f"SHAP Waterfall: High-Propensity Client (P={probas[high_idx]:.2f})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_waterfall_high_propensity.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_class1[low_idx], max_display=12, show=False)
    plt.title(f"SHAP Waterfall: Low-Propensity Client (P={probas[low_idx]:.2f})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_waterfall_low_propensity.png", dpi=300, bbox_inches="tight")
    plt.close()

    return shap_values_class1, explainer


def explain_single_instance(
    explainer: Any,
    X_single: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Returns top positive and negative feature contributions for an individual client
    for display in the Streamlit UI.
    """
    shap_val = explainer(X_single)
    if len(shap_val.shape) == 3:
        vals = shap_val.values[0, :, 1]
    elif len(shap_val.shape) == 2:
        vals = shap_val.values[0]
    else:
        vals = shap_val.values

    feat_vals = X_single[0]
    contributions = []
    for fn, val, sv in zip(feature_names, feat_vals, vals):
        contributions.append({
            "feature": fn,
            "feature_value": float(val),
            "shap_value": float(sv),
            "impact": "Increases Subscription Likelihood" if sv > 0 else "Decreases Subscription Likelihood"
        })

    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return contributions[:top_n]
