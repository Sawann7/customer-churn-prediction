"""
Strategic Insights Page - Actionable retention playbook and empirical drivers,
cleanly separated from scientific caveats and methodological limitations.
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.eda import compute_churn_by_categorical, compute_churn_by_tenure_groups
from src.ui_theme import apply_custom_theme, render_hero, render_theme_toggle

st.set_page_config(page_title="Strategic Retention Insights | Telco Churn", page_icon="💡", layout="wide")
render_theme_toggle()
apply_custom_theme()

render_hero(
    title="Strategic Retention Insights & Playbook",
    subtitle="Actionable, data-derived retention playbooks and governance principles synthesized from 7,043 customer accounts.",
    badge_text="Module 5: Strategy Playbook",
)


@st.cache_data
def get_data():
    return load_data()


try:
    df = get_data()

    # Dynamic metrics
    contract_df = compute_churn_by_categorical(df, "Contract")
    m2m_rate = contract_df.loc[contract_df["Contract"] == "Month-to-month", "churn_rate_pct"].values[0]
    two_yr_rate = contract_df.loc[contract_df["Contract"] == "Two year", "churn_rate_pct"].values[0]

    pay_df = compute_churn_by_categorical(df, "PaymentMethod")
    echeck_rate = pay_df.loc[pay_df["PaymentMethod"] == "Electronic check", "churn_rate_pct"].values[0]
    auto_rates = pay_df.loc[pay_df["PaymentMethod"].str.contains("automatic"), "churn_rate_pct"].mean()

    int_df = compute_churn_by_categorical(df, "InternetService")
    fiber_rate = int_df.loc[int_df["InternetService"] == "Fiber optic", "churn_rate_pct"].values[0]
    dsl_rate = int_df.loc[int_df["InternetService"] == "DSL", "churn_rate_pct"].values[0]

    tenure_df = compute_churn_by_tenure_groups(df)
    y1_rate = tenure_df.loc[tenure_df["tenure_group"] == "0-12 Months (Year 1)", "churn_rate_pct"].values[0]

    tab_playbook, tab_limitations = st.tabs(["💡 Executive Retention Playbook", "📖 Scientific Governance & Limitations"])

    # =========================================================================
    # TAB 1: EXECUTIVE RETENTION PLAYBOOK (ACTIONABLE MODE)
    # =========================================================================
    with tab_playbook:
        st.subheader("🎯 Priority Retention Opportunities")

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #2563eb; margin-bottom: 1rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1e3a8a;">
                        1. Contract Lock-in Incentives
                    </div>
                    <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.2rem; margin-bottom: 0.5rem;">
                        Empirical Gap: <b>{m2m_rate:.1f}%</b> (Month-to-month) vs <b>{two_yr_rate:.1f}%</b> (Two-year)
                    </div>
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.5;">
                        • <b>Action:</b> Target month-to-month accounts reaching month 6 with a 15% discount for 6 months in exchange for a 1-year agreement.<br>
                        • <b>Expected Impact:</b> Converts vulnerable month-to-month subscribers into sticky contracted accounts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #10b981; margin-bottom: 1rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #065f46;">
                        2. Automated Billing Credits
                    </div>
                    <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.2rem; margin-bottom: 0.5rem;">
                        Empirical Gap: <b>{echeck_rate:.1f}%</b> (Electronic Check) vs <b>{auto_rates:.1f}%</b> (Auto-pay)
                    </div>
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.5;">
                        • <b>Action:</b> Offer a recurring $5/month billing credit for switching from manual check to automated credit card/bank transfer.<br>
                        • <b>Expected Impact:</b> Eliminates monthly payment cognitive friction and lowers attrition towards the ~16% baseline.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row1_col2:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #f59e0b; margin-bottom: 1rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #92400e;">
                        3. Value-Added Service Bundles for Fiber
                    </div>
                    <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.2rem; margin-bottom: 0.5rem;">
                        Empirical Gap: Fiber churn is <b>{fiber_rate:.1f}%</b> vs <b>{dsl_rate:.1f}%</b> for DSL
                    </div>
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.5;">
                        • <b>Action:</b> Bundle complimentary Tech Support and Online Security into premium fiber plans.<br>
                        • <b>Expected Impact:</b> Accounts with Tech Support churn at only 15.2% (vs 41.6% without), mitigating price sensitivity.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #8b5cf6; margin-bottom: 1rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #5b21b6;">
                        4. 'First 90 Days' Customer Onboarding
                    </div>
                    <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.2rem; margin-bottom: 0.5rem;">
                        Empirical Gap: Year 1 churn is <b>{y1_rate:.1f}%</b>, declining to &lt; 10% after Year 4
                    </div>
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.5;">
                        • <b>Action:</b> Proactive customer success check-ins at Day 30, 60, and 90 to ensure setup satisfaction.<br>
                        • <b>Expected Impact:</b> Directly tackles the period of peak customer vulnerability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 2: SCIENTIFIC GOVERNANCE & LIMITATIONS (DOCUMENTATION MODE)
    # =========================================================================
    with tab_limitations:
        st.subheader("📖 Analytical Governance & Methodological Constraints")

        st.markdown(
            """
            <div class="doc-box">
                <div class="doc-title">Association vs. Proof of Causation</div>
                <div class="doc-text">
                    A customer with a month-to-month contract is <i>associated</i> with higher churn, but the contract itself may not <i>cause</i> the churn.
                    Customers who already suspect they might relocate or switch providers intentionally select month-to-month contracts.
                    Proposed retention interventions should always be piloted through controlled A/B tests rather than assuming changes will mirror historical distributions.
                </div>
            </div>

            <div class="doc-box">
                <div class="doc-title">Dataset Telemetry Constraints</div>
                <div class="doc-text">
                    The IBM Telco dataset is a point-in-time cross-sectional snapshot. It lacks continuous event logs such as:
                    <br>• Inbound customer support call frequency or complaint ticket logs
                    <br>• Real-time network speed drops or local outage durations
                    <br>• Regional competitor pricing and promotional marketing campaigns
                </div>
            </div>

            <div class="doc-box">
                <div class="doc-title">Survivorship Bias in Tenure Analysis</div>
                <div class="doc-text">
                    High-tenure cohorts (50+ months) naturally select for customers who are satisfied with the service.
                    Tenure is therefore both an indicator and consequence of historical customer satisfaction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"Error loading insights: {str(e)}")
