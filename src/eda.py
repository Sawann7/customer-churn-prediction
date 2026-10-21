"""
Exploratory Data Analysis (EDA) module for Telco Customer Churn.
Calculates real summary statistics, categorical breakdowns, tenure groupings,
correlations, and key data-driven findings.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_overall_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level customer and churn metrics directly from actual data.
    """
    total = len(df)
    if "Churn" not in df.columns:
        raise ValueError("Missing 'Churn' column in dataframe.")

    churn_counts = df["Churn"].astype(str).str.strip().value_counts()
    churned = int(churn_counts.get("Yes", churn_counts.get("1", 0)))
    retained = int(churn_counts.get("No", churn_counts.get("0", 0)))
    churn_rate = (churned / total * 100) if total > 0 else 0.0
    retention_rate = (retained / total * 100) if total > 0 else 0.0

    # Numerical metrics
    num_metrics = {}
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            num_metrics[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }

    return {
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate": churn_rate,
        "retention_rate": retention_rate,
        "numerical_metrics": num_metrics,
    }


def compute_churn_by_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Computes exact customer counts, churned counts, and churn percentages for categories in `col`.
    """
    if col not in df.columns or "Churn" not in df.columns:
        raise ValueError(f"Columns '{col}' or 'Churn' missing from dataframe.")

    temp = df[[col, "Churn"]].copy()
    temp["Churn_binary"] = temp["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0)

    grouped = temp.groupby(col).agg(
        total_customers=("Churn_binary", "count"),
        churned_customers=("Churn_binary", "sum"),
    ).reset_index()

    grouped["retained_customers"] = grouped["total_customers"] - grouped["churned_customers"]
    grouped["churn_rate_pct"] = (grouped["churned_customers"] / grouped["total_customers"] * 100).round(2)
    grouped["share_of_total_pct"] = (grouped["total_customers"] / len(temp) * 100).round(2)

    return grouped.sort_values(by="churn_rate_pct", ascending=False)


def compute_churn_by_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segments customers by tenure into operational tenure bands and calculates churn rate.
    """
    temp = df[["tenure", "Churn"]].copy()
    temp["tenure_num"] = pd.to_numeric(temp["tenure"], errors="coerce")
    temp["Churn_binary"] = temp["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0)

    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12 Months (Year 1)", "13-24 Months (Year 2)", "25-48 Months (Years 3-4)", "49-72 Months (Years 5-6)"]
    temp["tenure_group"] = pd.cut(temp["tenure_num"], bins=bins, labels=labels)

    grouped = temp.groupby("tenure_group", observed=False).agg(
        total_customers=("Churn_binary", "count"),
        churned_customers=("Churn_binary", "sum"),
    ).reset_index()

    grouped["retained_customers"] = grouped["total_customers"] - grouped["churned_customers"]
    grouped["churn_rate_pct"] = (grouped["churned_customers"] / grouped["total_customers"] * 100).round(2)
    return grouped


def compute_churn_by_monthly_charges_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segments customers by monthly charges into quartiles/tiers and calculates churn rate.
    """
    temp = df[["MonthlyCharges", "Churn"]].copy()
    temp["MonthlyCharges_num"] = pd.to_numeric(temp["MonthlyCharges"], errors="coerce")
    temp["Churn_binary"] = temp["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0)

    bins = [0, 35, 70, 95, 150]
    labels = ["Low (< $35)", "Medium ($35 - $70)", "High ($70 - $95)", "Very High (> $95)"]
    temp["charges_tier"] = pd.cut(temp["MonthlyCharges_num"], bins=bins, labels=labels)

    grouped = temp.groupby("charges_tier", observed=False).agg(
        total_customers=("Churn_binary", "count"),
        churned_customers=("Churn_binary", "sum"),
    ).reset_index()

    grouped["retained_customers"] = grouped["total_customers"] - grouped["churned_customers"]
    grouped["churn_rate_pct"] = (grouped["churned_customers"] / grouped["total_customers"] * 100).round(2)
    return grouped


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix for numerical features and binary churn.
    """
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    temp = df[cols].copy()
    for c in cols:
        temp[c] = pd.to_numeric(temp[c], errors="coerce")

    if "Churn" in df.columns:
        temp["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0)

    return temp.corr().round(3)


def generate_key_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes factual, data-derived insights from actual statistics.
    Every number is calculated directly from the dataset.
    Explicitly caveats all observations as statistical associations, not causal proofs.
    """
    insights = []

    # 1. Contract Analysis
    contract_df = compute_churn_by_categorical(df, "Contract")
    m2m = contract_df[contract_df["Contract"] == "Month-to-month"]
    two_yr = contract_df[contract_df["Contract"] == "Two year"]
    if not m2m.empty and not two_yr.empty:
        m2m_rate = m2m["churn_rate_pct"].values[0]
        two_yr_rate = two_yr["churn_rate_pct"].values[0]
        insights.append({
            "title": "Contract Type & Churn Association",
            "stat": f"{m2m_rate:.1f}% vs {two_yr_rate:.1f}%",
            "description": (
                f"Customers on Month-to-month contracts exhibit a churn rate of {m2m_rate:.1f}%, compared to just "
                f"{two_yr_rate:.1f}% for customers on Two-year contracts. Month-to-month subscribers represent "
                f"{m2m['total_customers'].values[0]} total accounts. This demonstrates a strong statistical association "
                "between contract commitment and retention, although contract type itself may reflect underlying customer intent."
            ),
            "category": "Contract"
        })

    # 2. Tenure Analysis
    tenure_df = compute_churn_by_tenure_groups(df)
    y1 = tenure_df[tenure_df["tenure_group"] == "0-12 Months (Year 1)"]
    y5_6 = tenure_df[tenure_df["tenure_group"] == "49-72 Months (Years 5-6)"]
    if not y1.empty and not y5_6.empty:
        y1_rate = y1["churn_rate_pct"].values[0]
        y5_6_rate = y5_6["churn_rate_pct"].values[0]
        insights.append({
            "title": "Onboarding & Tenure Sensitivity",
            "stat": f"{y1_rate:.1f}% vs {y5_6_rate:.1f}%",
            "description": (
                f"First-year customers (0-12 months tenure) experience the highest churn rate at {y1_rate:.1f}%, "
                f"whereas long-tenured customers (49-72 months) churn at only {y5_6_rate:.1f}%. "
                "The initial 12 months represent the period of highest attrition risk."
            ),
            "category": "Tenure"
        })

    # 3. Internet Service Analysis
    internet_df = compute_churn_by_categorical(df, "InternetService")
    fiber = internet_df[internet_df["InternetService"] == "Fiber optic"]
    dsl = internet_df[internet_df["InternetService"] == "DSL"]
    no_int = internet_df[internet_df["InternetService"] == "No"]
    if not fiber.empty and not dsl.empty:
        fiber_rate = fiber["churn_rate_pct"].values[0]
        dsl_rate = dsl["churn_rate_pct"].values[0]
        insights.append({
            "title": "Internet Service Type Disparity",
            "stat": f"{fiber_rate:.1f}% (Fiber) vs {dsl_rate:.1f}% (DSL)",
            "description": (
                f"Fiber optic internet users show an elevated churn rate of {fiber_rate:.1f}%, significantly higher than "
                f"DSL users ({dsl_rate:.1f}%) and customers without internet service ("
                f"{no_int['churn_rate_pct'].values[0] if not no_int.empty else 'N/A'}%). "
                "Fiber optic plans carry higher monthly bills and may face greater competition or service expectations."
            ),
            "category": "Service"
        })

    # 4. Payment Method
    pay_df = compute_churn_by_categorical(df, "PaymentMethod")
    e_check = pay_df[pay_df["PaymentMethod"] == "Electronic check"]
    if not e_check.empty:
        e_rate = e_check["churn_rate_pct"].values[0]
        other_avg = pay_df[pay_df["PaymentMethod"] != "Electronic check"]["churn_rate_pct"].mean()
        insights.append({
            "title": "Payment Method Friction",
            "stat": f"{e_rate:.1f}% (Electronic Check) vs {other_avg:.1f}% (Average of Others)",
            "description": (
                f"Customers paying via Electronic Check churn at {e_rate:.1f}%, nearly double the average "
                f"churn rate of automated payment methods ({other_avg:.1f}%). Automated bank transfer and credit "
                "card methods correlate with substantially higher customer retention."
            ),
            "category": "Billing"
        })

    # 5. Tech Support & Security Add-ons
    if "TechSupport" in df.columns:
        ts_df = compute_churn_by_categorical(df, "TechSupport")
        no_ts = ts_df[ts_df["TechSupport"] == "No"]
        yes_ts = ts_df[ts_df["TechSupport"] == "Yes"]
        if not no_ts.empty and not yes_ts.empty:
            no_ts_rate = no_ts["churn_rate_pct"].values[0]
            yes_ts_rate = yes_ts["churn_rate_pct"].values[0]
            insights.append({
                "title": "Protective & Support Services",
                "stat": f"{no_ts_rate:.1f}% (No Tech Support) vs {yes_ts_rate:.1f}% (With Support)",
                "description": (
                    f"Customers without Tech Support exhibit a churn rate of {no_ts_rate:.1f}%, compared to "
                    f"{yes_ts_rate:.1f}% for those who subscribe to Tech Support. Value-added services appear "
                    "associated with higher stickiness and switching costs."
                ),
                "category": "Support"
            })

    return insights


# ==========================================
# Matplotlib / Seaborn Visualizations
# ==========================================

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE_CHURN = {"No": "#2b5c8f", "Yes": "#d9534f", 0: "#2b5c8f", 1: "#d9534f"}


def plot_churn_distribution(df: pd.DataFrame) -> plt.Figure:
    """Creates a clean donut chart and count bar for overall churn."""
    counts = df["Churn"].astype(str).str.strip().value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    labels = ["Retained (No)", "Churned (Yes)"]
    colors = ["#2b5c8f", "#d9534f"]
    sizes = [counts.get("No", counts.get("0", 0)), counts.get("Yes", counts.get("1", 0))]

    # Donut chart
    wedges, texts, autotexts = axes[0].pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    axes[0].set_title("Overall Churn Proportion", fontsize=12, pad=10)

    # Bar chart
    bars = axes[1].bar(labels, sizes, color=colors, width=0.5, edgecolor="none")
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    axes[1].set_ylabel("Number of Customers", fontsize=10)
    axes[1].set_title("Customer Counts by Status", fontsize=12, pad=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    return fig


def plot_categorical_churn(df: pd.DataFrame, col: str, top_n: int = 10) -> plt.Figure:
    """Plots counts and churn rate side-by-side for a categorical column."""
    grouped = compute_churn_by_categorical(df, col).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Stacked count bar chart
    categories = grouped[col].astype(str)
    retained = grouped["retained_customers"]
    churned = grouped["churned_customers"]

    axes[0].bar(categories, retained, label="Retained (No)", color="#2b5c8f", alpha=0.85)
    axes[0].bar(categories, churned, bottom=retained, label="Churned (Yes)", color="#d9534f", alpha=0.9)
    axes[0].set_ylabel("Customer Count", fontsize=10)
    axes[0].set_title(f"Customer Distribution by {col}", fontsize=12, pad=10)
    axes[0].legend(loc="upper right")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Churn rate bar chart
    bars = axes[1].bar(categories, grouped["churn_rate_pct"], color="#d9534f", alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    axes[1].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[1].set_title(f"Churn Rate (%) by {col}", fontsize=12, pad=10)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_numerical_distributions(df: pd.DataFrame) -> plt.Figure:
    """Plots distribution boxplots and KDEs for tenure and monthly charges partitioned by churn."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    titles = ["Tenure (Months)", "Monthly Charges ($)", "Total Charges ($)"]

    for i, (col, title) in enumerate(zip(cols, titles)):
        temp = df[[col, "Churn"]].copy()
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
        temp = temp.dropna(subset=[col])
        temp["Churn"] = temp["Churn"].astype(str).str.strip().map({"Yes": "Churned", "No": "Retained"})

        sns.boxplot(
            data=temp,
            x="Churn",
            y=col,
            ax=axes[i],
            palette={"Retained": "#2b5c8f", "Churned": "#d9534f"},
            fliersize=2,
            width=0.45,
        )
        axes[i].set_title(f"{title} by Churn", fontsize=11, pad=8)
        axes[i].set_xlabel("")
        axes[i].set_ylabel(title, fontsize=10)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Plots correlation matrix heatmap."""
    corr = compute_correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(6, 4.8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Numerical & Target Correlation Heatmap", fontsize=12, pad=10)
    fig.tight_layout()
    return fig
