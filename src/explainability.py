"""
Model explainability module.
Extracts global feature importances, logistic regression coefficients,
permutation importances, and single-prediction factor breakdowns.
Explicitly frames findings as predictive associations, not causal proofs.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from src.preprocessing import get_feature_names


def get_global_feature_importance(pipeline: Any, top_n: int = 15) -> pd.DataFrame:
    """
    Extracts post-transformation feature importances or coefficients from a fitted pipeline.
    Returns a DataFrame sorted by absolute importance.
    """
    if "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError("Pipeline must have 'preprocessor' and 'classifier' named steps.")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor)

    df_imp = pd.DataFrame({"Feature": feature_names})

    if hasattr(classifier, "feature_importances_"):
        df_imp["Importance"] = classifier.feature_importances_
        df_imp["Abs_Importance"] = df_imp["Importance"].abs()
        df_imp["Direction"] = "Non-linear / Tree Split"
        df_imp["Metric"] = "Gini / Split Gain"
    elif hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]
        df_imp["Importance"] = coefs
        df_imp["Abs_Importance"] = np.abs(coefs)
        df_imp["Direction"] = np.where(coefs > 0, "Increases Churn Risk", "Decreases Churn Risk")
        df_imp["Metric"] = "Log-Odds Coefficient"
    else:
        raise AttributeError("Classifier does not expose feature_importances_ or coef_.")

    df_imp = df_imp.sort_values(by="Abs_Importance", ascending=False).reset_index(drop=True)
    return df_imp.head(top_n)


def compute_permutation_importance(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Computes model-agnostic permutation feature importance on validation/test data.
    """
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc"
    )

    df_perm = pd.DataFrame({
        "Feature": X_val.columns,
        "Importance": result.importances_mean,
        "Importance_Std": result.importances_std,
        "Abs_Importance": np.abs(result.importances_mean),
    }).sort_values(by="Abs_Importance", ascending=False).reset_index(drop=True)

    return df_perm.head(top_n)


def explain_single_prediction(
    pipeline: Any,
    customer_df: pd.DataFrame,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Provides an interpretable heuristic breakdown of the key factors driving this specific
    customer's churn prediction, based on model feature weights and input values.
    Explicitly labeled as model-based associative attribution.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if not preprocessor or not classifier:
        return []

    # Transform single row
    transformed = preprocessor.transform(customer_df)
    feature_names = get_feature_names(preprocessor)

    factors = []
    if hasattr(classifier, "coef_"):
        # For linear models: contribution is value * coefficient
        coefs = classifier.coef_[0]
        contributions = transformed[0] * coefs
        for feat, val, coef, cont in zip(feature_names, transformed[0], coefs, contributions):
            if abs(cont) > 1e-4:
                factors.append({
                    "feature": feat,
                    "impact_score": float(cont),
                    "abs_impact": float(abs(cont)),
                    "direction": "Risk Driver" if cont > 0 else "Protective Factor",
                    "explanation": (
                        f"Increases churn probability (weight: +{cont:.2f})"
                        if cont > 0
                        else f"Lowers churn probability (weight: {cont:.2f})"
                    ),
                })
    elif hasattr(classifier, "feature_importances_"):
        # For tree models: examine non-zero transformed values weighted by tree importance
        importances = classifier.feature_importances_
        contributions = transformed[0] * importances
        for feat, val, imp, cont in zip(feature_names, transformed[0], importances, contributions):
            if abs(val) > 1e-4 and imp > 1e-4:
                factors.append({
                    "feature": feat,
                    "impact_score": float(cont),
                    "abs_impact": float(abs(cont)),
                    "direction": "Key Influencer",
                    "explanation": f"Important predictive attribute in decision trees (importance: {imp:.3f})",
                })

    factors.sort(key=lambda x: x["abs_impact"], reverse=True)
    return factors[:top_n]


def plot_feature_importances(df_imp: pd.DataFrame, title: Optional[str] = None) -> plt.Figure:
    """
    Renders horizontal bar chart of top feature importances.
    """
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(df_imp) * 0.35)))

    # Directional colors if available
    if "Direction" in df_imp.columns and "Increases Churn Risk" in df_imp["Direction"].values:
        palette = {
            "Increases Churn Risk": "#d9534f",
            "Decreases Churn Risk": "#2b5c8f",
            "Non-linear / Tree Split": "#6c757d",
        }
        sns.barplot(
            data=df_imp,
            x="Importance",
            y="Feature",
            hue="Direction",
            palette=palette,
            ax=ax,
            edgecolor="none"
        )
    else:
        sns.barplot(
            data=df_imp,
            x="Abs_Importance",
            y="Feature",
            color="#2b5c8f",
            ax=ax,
            edgecolor="none"
        )

    chart_title = title or "Top Predictive Features (Model Association)"
    ax.set_title(chart_title, fontsize=12, pad=10)
    ax.set_xlabel("Relative Importance / Influence", fontsize=10)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
