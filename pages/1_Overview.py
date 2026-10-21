"""
Overview Page - High-level dataset summary, interactive churn distributions,
and correlation analysis cleanly separated from detailed background text.
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.eda import compute_overall_summary, generate_key_insights
from src.interactive_plots import (
    plot_churn_donut_plotly,
    plot_correlation_heatmap_plotly,
)
from src.ui_theme import apply_custom_theme, render_hero, render_theme_toggle

st.set_page_config(page_title="Executive Overview | Telco Churn", page_icon="📊", layout="wide")
render_theme_toggle()
apply_custom_theme()

render_hero(
    title="Executive Overview & Churn Distribution",
    subtitle="High-level retention metrics, interactive churn proportions, and numerical feature correlations.",
    badge_text="Module 1: Global Health",
)


@st.cache_data
def get_data():
    return load_data()


try:
    df = get_data()
    summary = compute_overall_summary(df)
    insights = generate_key_insights(df)

    # Clean separation between Active Dashboard and Detailed Documentation
    tab_dashboard, tab_details = st.tabs(["📊 Interactive Visual Dashboard", "📖 Detailed Observations & Statistical Notes"])

    # =========================================================================
    # TAB 1: INTERACTIVE VISUAL DASHBOARD (WORKING MODE)
    # =========================================================================
    with tab_dashboard:
        # Top KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Total Accounts</div>
                    <div class="card-number">{summary['total_customers']:,}</div>
                    <div class="card-delta delta-positive">Full active dataset</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Retained Cohort</div>
                    <div class="card-number">{summary['retained_customers']:,}</div>
                    <div class="card-delta delta-positive">+{summary['retention_rate']:.1f}% stable customers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Churned Cohort</div>
                    <div class="card-number">{summary['churned_customers']:,}</div>
                    <div class="card-delta delta-negative">-{summary['churn_rate']:.1f}% baseline churn</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            avg_t = summary["numerical_metrics"]["tenure"]["mean"]
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Average Tenure</div>
                    <div class="card-number">{avg_t:.1f} mo</div>
                    <div class="card-delta delta-positive">Median: {summary['numerical_metrics']['tenure']['median']:.0f} mo</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Visualizations (Plotly Interactive)
        c_vis1, c_vis2 = st.columns([1, 1])
        with c_vis1:
            fig_donut = plot_churn_donut_plotly(df)
            st.plotly_chart(fig_donut, use_container_width=True)

        with c_vis2:
            fig_corr = plot_correlation_heatmap_plotly(df)
            st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()

        # Summary Statistics Table
        st.subheader("📋 Numerical Features Summary Statistics")
        stats_data = []
        for col_name, m in summary["numerical_metrics"].items():
            stats_data.append({
                "Feature": col_name,
                "Mean": f"{m['mean']:.2f}",
                "Median": f"{m['median']:.2f}",
                "Std Dev": f"{m['std']:.2f}",
                "Min": f"{m['min']:.2f}",
                "Max": f"{m['max']:.2f}",
            })
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: DETAILED OBSERVATIONS & STATISTICAL NOTES (DOCUMENTATION MODE)
    # =========================================================================
    with tab_details:
        st.subheader("💡 Key Statistical Observations (Empirical Findings)")
        st.caption(
            "Every metric below is computed directly from customer records and represents "
            "statistical associations, not causal proofs."
        )

        for ins in insights:
            st.markdown(
                f"""
                <div class="doc-box">
                    <div class="doc-title">{ins['title']} &mdash; <span style="color:#2563eb;">{ins['stat']}</span></div>
                    <div class="doc-text">{ins['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

except Exception as e:
    st.error(f"Error loading overview data: {str(e)}")
