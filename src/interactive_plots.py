"""
Interactive Plotly visualizations for Telco Customer Churn.
Provides interactive charts with responsive hover tooltips, zoom, custom palettes,
and dynamic Dark/Light theme adaptation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.eda import (
    compute_churn_by_categorical,
    compute_churn_by_monthly_charges_bins,
    compute_churn_by_tenure_groups,
    compute_correlation_matrix,
)
from src.ui_theme import get_current_theme

PALETTE = {
    "Retained": "#2563eb",  # Modern Cobalt Blue
    "Churned": "#ef4444",   # Modern Coral Red
    "Neutral": "#64748b",   # Slate
    "Accent": "#06b6d4",    # Cyan
    "Success": "#10b981",   # Emerald Green
    "Warning": "#f59e0b",   # Amber
}


def _get_plotly_template() -> Dict[str, Any]:
    theme = get_current_theme()
    if theme == "dark":
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font_color": "#e2e8f0",
            "grid_color": "#1e293b",
        }
    return {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_color": "#1e293b",
        "grid_color": "#f1f5f9",
    }


def plot_churn_donut_plotly(df: pd.DataFrame) -> go.Figure:
    """Creates a sleek, interactive donut chart showing customer retention vs churn."""
    cfg = _get_plotly_template()
    counts = df["Churn"].astype(str).str.strip().value_counts()
    retained = int(counts.get("No", counts.get("0", 0)))
    churned = int(counts.get("Yes", counts.get("1", 0)))
    total = retained + churned

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Retained (No)", "Churned (Yes)"],
                values=[retained, churned],
                hole=0.6,
                marker=dict(colors=[PALETTE["Retained"], PALETTE["Churned"]]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Customer Count: %{value:,}<br>Share: %{percent}<extra></extra>",
                textfont=dict(size=14, color="#ffffff"),
            )
        ]
    )
    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        title=dict(text="<b>Overall Churn Distribution</b>", x=0.5, font=dict(size=16, color=cfg["font_color"])),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br><span style='font-size:12px;color:#94a3b8;'>Accounts</span>",
                x=0.5,
                y=0.5,
                font_size=18,
                showarrow=False,
            )
        ],
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=340,
    )
    return fig


def plot_categorical_churn_plotly(df: pd.DataFrame, col: str, title: Optional[str] = None) -> go.Figure:
    """Creates an interactive dual-axis bar & line chart showing volume and churn rate."""
    cfg = _get_plotly_template()
    grouped = compute_churn_by_categorical(df, col)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=grouped[col],
            y=grouped["retained_customers"],
            name="Retained",
            marker_color=PALETTE["Retained"],
            hovertemplate="<b>%{x}</b><br>Retained: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped[col],
            y=grouped["churned_customers"],
            name="Churned",
            marker_color=PALETTE["Churned"],
            hovertemplate="<b>%{x}</b><br>Churned: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=grouped[col],
            y=grouped["churn_rate_pct"],
            name="Churn Rate (%)",
            mode="lines+markers+text",
            marker=dict(size=9, color="#f59e0b"),
            line=dict(color="#f59e0b", width=3, dash="dot"),
            yaxis="y2",
            text=[f"{v:.1f}%" for v in grouped["churn_rate_pct"]],
            textposition="top center",
            textfont=dict(size=11, color="#f59e0b", weight="bold"),
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        barmode="stack",
        title=dict(text=f"<b>{title or f'Churn Analysis by {col}'}</b>", x=0.0, font=dict(size=15, color=cfg["font_color"])),
        xaxis=dict(title=col, tickangle=-20 if len(grouped) > 3 else 0, showgrid=False),
        yaxis=dict(title="Customer Count", showgrid=True, gridcolor=cfg["grid_color"]),
        yaxis2=dict(
            title="Churn Rate (%)",
            overlaying="y",
            side="right",
            range=[0, max(grouped["churn_rate_pct"].max() * 1.25, 60)],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def plot_correlation_heatmap_plotly(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation matrix heatmap."""
    cfg = _get_plotly_template()
    corr = compute_correlation_matrix(df)
    labels = list(corr.columns)
    z = corr.values

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(z, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate="Feature 1: %{x}<br>Feature 2: %{y}<br>Pearson Corr: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        title=dict(text="<b>Numerical Feature Correlation Matrix</b>", x=0.0, font=dict(size=15, color=cfg["font_color"])),
        margin=dict(l=30, r=30, t=50, b=30),
        height=360,
    )
    return fig


def plot_numerical_box_plotly(df: pd.DataFrame, num_col: str) -> go.Figure:
    """Interactive box/violin plot comparing numerical distributions by churn status."""
    cfg = _get_plotly_template()
    temp = df[[num_col, "Churn"]].copy()
    temp[num_col] = pd.to_numeric(temp[num_col], errors="coerce")
    temp = temp.dropna(subset=[num_col])
    temp["Status"] = temp["Churn"].astype(str).str.strip().map({"Yes": "Churned", "No": "Retained"})

    fig = px.box(
        temp,
        x="Status",
        y=num_col,
        color="Status",
        color_discrete_map={"Retained": PALETTE["Retained"], "Churned": PALETTE["Churned"]},
        points="outliers",
        title=f"<b>Distribution of {num_col} by Churn Status</b>",
    )
    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        showlegend=False,
        margin=dict(l=30, r=30, t=50, b=30),
        yaxis=dict(showgrid=True, gridcolor=cfg["grid_color"]),
        height=340,
    )
    return fig


def plot_metrics_comparison_plotly(all_evals: List[Dict[str, Any]]) -> go.Figure:
    """Interactive multi-metric comparison grouped bar chart."""
    cfg = _get_plotly_template()
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall (Catch Rate)", "F1-Score", "ROC-AUC"]

    fig = go.Figure()
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    for i, res in enumerate(all_evals):
        scores = [res.get(m, 0.0) or 0.0 for m in metrics]
        fig.add_trace(
            go.Bar(
                name=res["model_name"],
                x=metric_labels,
                y=scores,
                marker_color=colors[i % len(colors)],
                text=[f"{s:.2f}" for s in scores],
                textposition="outside",
                hovertemplate=f"<b>{res['model_name']}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        barmode="group",
        title=dict(text="<b>Cross-Model Benchmark (Held-out Test Partition)</b>", x=0.0, font=dict(size=16, color=cfg["font_color"])),
        yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.15], showgrid=True, gridcolor=cfg["grid_color"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=30),
        height=380,
    )
    return fig


def plot_roc_curves_plotly(all_evals: List[Dict[str, Any]]) -> go.Figure:
    """Interactive multi-model ROC Curves."""
    cfg = _get_plotly_template()
    fig = go.Figure()
    colors = ["#2563eb", "#10b981", "#ef4444", "#8b5cf6"]

    for i, res in enumerate(all_evals):
        roc = res.get("roc_data")
        if roc:
            fig.add_trace(
                go.Scatter(
                    x=roc["fpr"],
                    y=roc["tpr"],
                    mode="lines",
                    name=f"{res['model_name']} (AUC = {roc.get('auc', 0):.3f})",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    hovertemplate=f"<b>{res['model_name']}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline Guess (AUC = 0.50)",
            line=dict(color="#64748b", dash="dash", width=1.5),
            hoverinfo="none",
        )
    )

    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        title=dict(text="<b>ROC Curves Comparison</b>", x=0.0, font=dict(size=15, color=cfg["font_color"])),
        xaxis=dict(title="False Positive Rate", range=[0, 1], showgrid=True, gridcolor=cfg["grid_color"]),
        yaxis=dict(title="True Positive Rate (Recall)", range=[0, 1.02], showgrid=True, gridcolor=cfg["grid_color"]),
        legend=dict(orientation="v", yanchor="bottom", y=0.05, xanchor="right", x=0.98),
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
    )
    return fig


def plot_churn_gauge_plotly(probability: float, decision_threshold: float = 0.50) -> go.Figure:
    """High-end interactive speedometer/gauge dial for single customer prediction."""
    cfg = _get_plotly_template()
    pct = round(probability * 100, 1)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": cfg["font_color"], "weight": "bold"}},
            title={"text": "<b>Estimated Churn Likelihood</b>", "font": {"size": 16, "color": cfg["font_color"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#3b82f6", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "#475569",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [35, 60], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [60, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.75,
                    "value": decision_threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        margin=dict(l=30, r=30, t=40, b=20),
        height=260,
    )
    return fig


def plot_feature_importances_plotly(df_imp: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Interactive horizontal bar chart of top model features."""
    cfg = _get_plotly_template()
    top_df = df_imp.head(top_n).sort_values(by="Abs_Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_df["Abs_Importance"],
            y=top_df["Feature"],
            orientation="h",
            marker=dict(
                color=top_df["Abs_Importance"],
                colorscale="Blues",
                line=dict(color="#1e3a8a", width=0.5),
            ),
            text=[f"{v:.3f}" for v in top_df["Abs_Importance"]],
            textposition="inside",
            insidetextanchor="end",
            hovertemplate="<b>%{y}</b><br>Relative Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=cfg["template"],
        paper_bgcolor=cfg["paper_bgcolor"],
        plot_bgcolor=cfg["plot_bgcolor"],
        font=dict(color=cfg["font_color"]),
        title=dict(text=f"<b>Top {top_n} Predictive Drivers</b>", x=0.0, font=dict(size=15, color=cfg["font_color"])),
        xaxis=dict(title="Importance Weight", showgrid=True, gridcolor=cfg["grid_color"]),
        yaxis=dict(title=""),
        margin=dict(l=140, r=30, t=50, b=30),
        height=380,
    )
    return fig
