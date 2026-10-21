"""
Predict Customer Churn Page - Live interactive prediction studio with profile presets,
Plotly speedometer gauge, local factor attribution, cleanly separated from documentation and disclaimers.
"""

import streamlit as st
import pandas as pd
from src.explainability import explain_single_prediction
from src.interactive_plots import plot_churn_gauge_plotly
from src.predict import (
    SAMPLE_PROFILES,
    classify_risk_tier,
    predict_churn,
    validate_customer_input,
)
from src.train import load_best_model, load_model_metadata
from src.ui_theme import apply_custom_theme, render_hero, render_theme_toggle

st.set_page_config(page_title="Predict Customer Churn | Telco Churn", page_icon="🔮", layout="wide")
render_theme_toggle()
apply_custom_theme()

render_hero(
    title="Predict Individual Customer Churn",
    subtitle="Interactive prediction studio: Enter customer attributes to estimate churn probability, view risk dial, and inspect local predictive drivers.",
    badge_text="Module 4: Prediction Studio",
)

pipeline = load_best_model()
meta = load_model_metadata()

if pipeline is None:
    st.warning("⚠️ No pre-trained model artifact was detected on disk.")
    if st.button("🚀 Train & Initialize Model Now (~10 seconds)"):
        with st.spinner("Training candidate pipelines and calibrating best model..."):
            from src.train import save_trained_artifacts, train_and_compare_models
            best_name, best_pipe, evals, meta = train_and_compare_models()
            save_trained_artifacts(best_pipe, meta, evals)
            st.success(f"Successfully trained and selected {best_name}! Refreshing...")
            st.rerun()
    st.stop()

tab_predictor, tab_guide = st.tabs(["⚡ Live Prediction Studio", "📖 Model Assumptions & Risk Threshold Guide"])

# =========================================================================
# TAB 1: LIVE PREDICTION STUDIO (WORKING MODE)
# =========================================================================
with tab_predictor:
    # Quick Preset Selector
    st.markdown("##### ⚡ Load Benchmark Profile Presets")
    preset_cols = st.columns(4)

    preset_choice = st.session_state.get("selected_preset", "High Risk Profile")

    with preset_cols[0]:
        if st.button("🚨 High Risk Customer", use_container_width=True):
            st.session_state["selected_preset"] = "High Risk Profile"
            preset_choice = "High Risk Profile"
    with preset_cols[1]:
        if st.button("🛡️ Low Risk Customer", use_container_width=True):
            st.session_state["selected_preset"] = "Low Risk Profile"
            preset_choice = "Low Risk Profile"
    with preset_cols[2]:
        if st.button("⚖️ Average Profile", use_container_width=True):
            st.session_state["selected_preset"] = "Average / Moderate Profile"
            preset_choice = "Average / Moderate Profile"
    with preset_cols[3]:
        if st.button("🔄 Blank Custom Form", use_container_width=True):
            st.session_state["selected_preset"] = "Custom / Blank"
            preset_choice = "Custom / Blank"

    default_vals = {}
    if preset_choice in SAMPLE_PROFILES:
        default_vals = SAMPLE_PROFILES[preset_choice]

    st.write("")

    # Clean, intuitive form
    with st.form("customer_intake_form"):
        col_sec1, col_sec2, col_sec3 = st.columns(3)

        with col_sec1:
            st.markdown("###### 👤 Account & Demographics")
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0 if default_vals.get("gender", "Male") == "Male" else 1,
            )
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes (Senior)" if x == 1 else "No",
                index=0 if default_vals.get("SeniorCitizen", 0) == 0 else 1,
            )
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
                index=0 if default_vals.get("Partner", "No") == "Yes" else 1,
            )
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"],
                index=0 if default_vals.get("Dependents", "No") == "Yes" else 1,
            )
            tenure = st.number_input(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(default_vals.get("tenure", 12)),
                step=1,
            )

        with col_sec2:
            st.markdown("###### 📡 Subscribed Services")
            phone_svc = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
                index=0 if default_vals.get("PhoneService", "Yes") == "Yes" else 1,
            )
            internet_opts = ["Fiber optic", "DSL", "No"]
            def_int = default_vals.get("InternetService", "Fiber optic")
            internet_svc = st.selectbox(
                "Internet Service",
                internet_opts,
                index=internet_opts.index(def_int) if def_int in internet_opts else 0,
            )
            sec_opts = ["No", "Yes", "No internet service"]
            def_sec = default_vals.get("OnlineSecurity", "No")
            online_sec = st.selectbox(
                "Online Security",
                sec_opts,
                index=sec_opts.index(def_sec) if def_sec in sec_opts else 0,
            )
            def_ts = default_vals.get("TechSupport", "No")
            tech_supp = st.selectbox(
                "Tech Support",
                sec_opts,
                index=sec_opts.index(def_ts) if def_ts in sec_opts else 0,
            )
            def_tv = default_vals.get("StreamingTV", "No")
            stream_tv = st.selectbox(
                "Streaming TV",
                sec_opts,
                index=sec_opts.index(def_tv) if def_tv in sec_opts else 0,
            )

        with col_sec3:
            st.markdown("###### 💳 Contract & Billing")
            contract_opts = ["Month-to-month", "One year", "Two year"]
            def_cntr = default_vals.get("Contract", "Month-to-month")
            contract = st.selectbox(
                "Contract Type",
                contract_opts,
                index=contract_opts.index(def_cntr) if def_cntr in contract_opts else 0,
            )
            pm_opts = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            def_pm = default_vals.get("PaymentMethod", "Electronic check")
            payment_method = st.selectbox(
                "Payment Method",
                pm_opts,
                index=pm_opts.index(def_pm) if def_pm in pm_opts else 0,
            )
            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=0 if default_vals.get("PaperlessBilling", "Yes") == "Yes" else 1,
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=150.0,
                value=float(default_vals.get("MonthlyCharges", 70.0)),
                step=1.0,
            )
            calc_tot = default_vals.get("TotalCharges")
            if calc_tot is None or pd.isna(calc_tot):
                calc_tot = round(monthly_charges * tenure, 2)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(calc_tot),
                step=25.0,
            )

        st.write("")
        col_thresh1, col_thresh2 = st.columns(2)
        with col_thresh1:
            decision_thresh = st.slider(
                "Decision Threshold (Classifies as Churn if prob >= threshold):",
                0.20, 0.80, 0.50, 0.05
            )
        with col_thresh2:
            high_risk_thresh = st.slider(
                "Operational High Risk Cutoff:",
                0.40, 0.80, 0.60, 0.05
            )

        submit_btn = st.form_submit_button("⚡ Run Churn Risk Assessment", use_container_width=True)

    # Customer profile dict
    cust_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_svc,
        "MultipleLines": default_vals.get("MultipleLines", "No"),
        "InternetService": internet_svc,
        "OnlineSecurity": online_sec,
        "OnlineBackup": default_vals.get("OnlineBackup", "No"),
        "DeviceProtection": default_vals.get("DeviceProtection", "No"),
        "TechSupport": tech_supp,
        "StreamingTV": stream_tv,
        "StreamingMovies": default_vals.get("StreamingMovies", "No"),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Run Prediction (automatic or on submit)
    thresholds = {"low_moderate": 0.35, "moderate_high": high_risk_thresh}
    res = predict_churn(
        cust_data,
        pipeline=pipeline,
        thresholds=thresholds,
        decision_threshold=decision_thresh,
    )

    prob = res["churn_probability"]
    label = res["predicted_label"]
    tier = res["risk_tier"]
    color = res["risk_color"]

    st.divider()

    # Visual Assessment Row (Interactive Plotly Gauge)
    st.subheader("🎯 Real-Time Churn Risk Assessment")
    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        fig_gauge = plot_churn_gauge_plotly(prob, decision_threshold=decision_thresh)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.markdown(
            f"""
            <div style="background-color: {color}15; border: 2px solid {color}; border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1.5rem;">
                <div style="font-size: 0.85rem; color: #64748b; font-weight: 700; letter-spacing: 0.05em;">ASSIGNED OPERATIONAL RISK TIER</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {color}; margin: 0.3rem 0;">{tier}</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b;">{label}</div>
                <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.6rem;">
                    Active Model: <b>{meta.get('model_name', 'Random Forest')}</b> &nbsp;|&nbsp; 
                    Decision Threshold: <b>{decision_thresh:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Local Factor Attribution
    st.write("")
    st.markdown("##### 🔍 Top Model Drivers for This Profile")
    cust_df = validate_customer_input(cust_data)
    factors = explain_single_prediction(pipeline, cust_df, top_n=5)

    if factors:
        factor_cards = []
        for f in factors:
            factor_cards.append({
                "Attribute": f["feature"],
                "Contribution Role": f["direction"],
                "Impact Score": f"{f['abs_impact']:.3f}",
                "Interpretation": f["explanation"],
            })
        st.dataframe(pd.DataFrame(factor_cards), use_container_width=True, hide_index=True)

# =========================================================================
# TAB 2: MODEL ASSUMPTIONS & RISK THRESHOLD GUIDE (DOCUMENTATION MODE)
# =========================================================================
with tab_guide:
    st.subheader("📖 Model Assumptions & Governance Guide")

    st.markdown(
        r"""
        <div class="doc-box">
            <div class="doc-title">How Predictions Are Computed</div>
            <div class="doc-text">
                1. <b>Pipeline Transformation:</b> Customer attributes undergo the exact preprocessing pipeline (numerical median imputation, scaling, and categorical one-hot encoding).<br>
                2. <b>Probability Estimation:</b> The calibrated classifier computes a continuous probability score \(P(\text{Churn})\) between 0.0 and 1.0.<br>
                3. <b>Decision Thresholding:</b> If \(P(\text{Churn}) \ge \text{Decision Threshold}\) (default 0.50), the customer is classified as 'Likely to Churn'.
            </div>
        </div>

        <div class="doc-box">
            <div class="doc-title">Operational Risk Tiers</div>
            <div class="doc-text">
                • <b>Low Risk (&lt; 35%):</b> Standard service and automated communications.<br>
                • <b>Moderate Risk (35% &ndash; 60%):</b> Eligible for courtesy check-in calls or product satisfaction surveys.<br>
                • <b>High Risk (&gt; 60%):</b> Priority candidate for proactive retention intervention (contract incentives, loyalty credits).
            </div>
        </div>

        <div class="doc-box">
            <div class="doc-title">Ethical Disclaimer & Model Scope</div>
            <div class="doc-text">
                This prediction reflects historical statistical associations and is not a certainty or guarantee of customer departure.
                Retention outreach should be paired with human customer service evaluation and company policy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
