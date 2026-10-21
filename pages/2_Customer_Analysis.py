"""
Customer Analysis Page - Interactive cohort segmentation, multi-attribute filtering,
and tenure patterns cleanly separated from methodology and descriptive text.
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.interactive_plots import (
    plot_categorical_churn_plotly,
    plot_numerical_box_plotly,
)
from src.ui_theme import apply_custom_theme, render_hero, render_theme_toggle

st.set_page_config(page_title="Customer Cohort Analysis | Telco Churn", page_icon="🔍", layout="wide")
render_theme_toggle()
apply_custom_theme()

render_hero(
    title="Customer Cohort Analysis & Segmentation",
    subtitle="Filter and compare customer attrition across contracts, payment methods, tenure cohorts, and service packages.",
    badge_text="Module 2: Cohort Explorer",
)


@st.cache_data
def get_data():
    return load_data()


try:
    df_raw = get_data()

    tab_explorer, tab_methodology = st.tabs(["⚡ Interactive Cohort Explorer", "📖 Cohort Definitions & Analysis Details"])

    # =========================================================================
    # TAB 1: INTERACTIVE COHORT EXPLORER (WORKING MODE)
    # =========================================================================
    with tab_explorer:
        # Dynamic Filter Bar in an Expander/Row
        st.markdown("##### 🎛️ Dynamic Cohort Filters")
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            contracts = ["All Contracts"] + sorted(df_raw["Contract"].unique().tolist())
            selected_contract = st.selectbox("Contract Type", contracts)

        with f2:
            internet_services = ["All Internet Types"] + sorted(df_raw["InternetService"].unique().tolist())
            selected_internet = st.selectbox("Internet Service", internet_services)

        with f3:
            payment_methods = ["All Payment Methods"] + sorted(df_raw["PaymentMethod"].unique().tolist())
            selected_payment = st.selectbox("Payment Method", payment_methods)

        with f4:
            senior_opts = ["All Customers", "Non-Senior (0)", "Senior Citizen (1)"]
            selected_senior = st.selectbox("Senior Status", senior_opts)

        # Apply active filters
        filtered_df = df_raw.copy()
        if selected_contract != "All Contracts":
            filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]
        if selected_internet != "All Internet Types":
            filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]
        if selected_payment != "All Payment Methods":
            filtered_df = filtered_df[filtered_df["PaymentMethod"] == selected_payment]
        if selected_senior == "Senior Citizen (1)":
            filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 1]
        elif selected_senior == "Non-Senior (0)":
            filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 0]

        # Live Filtered KPI Cards
        baseline_churn = (df_raw["Churn"] == "Yes").mean() * 100
        filtered_count = len(filtered_df)
        filtered_churn = (filtered_df["Churn"] == "Yes").mean() * 100 if filtered_count > 0 else 0.0
        diff = filtered_churn - baseline_churn

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            share = (filtered_count / len(df_raw) * 100) if len(df_raw) > 0 else 0
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Filtered Cohort</div>
                    <div class="card-number">{filtered_count:,}</div>
                    <div class="card-delta delta-positive">{share:.1f}% of total accounts</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            delta_class = "delta-negative" if diff > 0 else "delta-positive"
            delta_prefix = "+" if diff > 0 else ""
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Cohort Churn Rate</div>
                    <div class="card-number">{filtered_churn:.1f}%</div>
                    <div class="card-delta {delta_class}">{delta_prefix}{diff:.1f}% vs baseline ({baseline_churn:.1f}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c3:
            avg_m = filtered_df["MonthlyCharges"].mean() if filtered_count > 0 else 0
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Avg Monthly Bill</div>
                    <div class="card-number">${avg_m:.2f}</div>
                    <div class="card-delta delta-positive">Cohort average</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c4:
            avg_t = filtered_df["tenure"].mean() if filtered_count > 0 else 0
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">Avg Tenure</div>
                    <div class="card-number">{avg_t:.1f} mo</div>
                    <div class="card-delta delta-positive">Cohort average</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Visual Breakdown Sub-tabs (Plotly Interactive)
        st.subheader("📊 Interactive Attribute Distributions")
        vtab1, vtab2, vtab3 = st.tabs(["Contract & Payment", "Add-on Services", "Numerical Distributions"])

        with vtab1:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_cntr = plot_categorical_churn_plotly(df_raw, "Contract", title="Attrition by Contract Type")
                st.plotly_chart(fig_cntr, use_container_width=True)
            with col_b2:
                fig_pm = plot_categorical_churn_plotly(df_raw, "PaymentMethod", title="Attrition by Payment Method")
                st.plotly_chart(fig_pm, use_container_width=True)

        with vtab2:
            svc_col = st.selectbox(
                "Select Add-on Service to Inspect:",
                ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "InternetService"],
            )
            fig_svc = plot_categorical_churn_plotly(df_raw, svc_col, title=f"Attrition by {svc_col}")
            st.plotly_chart(fig_svc, use_container_width=True)

        with vtab3:
            col_bx1, col_bx2 = st.columns(2)
            with col_bx1:
                fig_box_t = plot_numerical_box_plotly(df_raw, "tenure")
                st.plotly_chart(fig_box_t, use_container_width=True)
            with col_bx2:
                fig_box_m = plot_numerical_box_plotly(df_raw, "MonthlyCharges")
                st.plotly_chart(fig_box_m, use_container_width=True)

        st.divider()

        # Data Inspector and Export
        st.subheader("📋 Cohort Raw Data Preview")
        st.dataframe(filtered_df.head(50), use_container_width=True)
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Current Filtered Cohort as CSV",
            data=csv_data,
            file_name="telco_filtered_cohort.csv",
            mime="text/csv",
        )

    # =========================================================================
    # TAB 2: COHORT DEFINITIONS & ANALYSIS DETAILS (DOCUMENTATION MODE)
    # =========================================================================
    with tab_methodology:
        st.subheader("📖 Cohort Definitions & Analytical Context")

        st.markdown(
            """
            <div class="doc-box">
                <div class="doc-title">Contract Sensitivity Analysis</div>
                <div class="doc-text">
                    • <b>Month-to-month contracts:</b> Suffer the highest attrition rate (42.7%), representing 3,875 of 7,043 total accounts.<br>
                    • <b>Two-year contracts:</b> Have a churn rate of only 2.8% (1,695 accounts). Long-term contracts establish clear structural barriers against impulsive switching.
                </div>
            </div>

            <div class="doc-box">
                <div class="doc-title">Payment Friction & Billing Psychology</div>
                <div class="doc-text">
                    • <b>Electronic Check:</b> Demonstrates a 45.3% churn rate. Manual payment requires active monthly customer interaction, prompting recurrent evaluation of the cost-benefit ratio.<br>
                    • <b>Automated Methods (Bank Transfer / Credit Card):</b> Experience average churn of ~16%. Automation creates friction-free subscription continuity.
                </div>
            </div>

            <div class="doc-box">
                <div class="doc-title">Support & Security Value Bundles</div>
                <div class="doc-text">
                    Subscribers lacking value-added security (Online Security: 41.8% churn, Tech Support: 41.6% churn) leave at nearly triple the rate of accounts with active protection (~15%).
                    These services create switching costs and increase overall product utility.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Error loading customer analysis: {str(e)}")
