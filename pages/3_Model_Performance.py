"""
Model Performance Page - Interactive multi-model benchmark, ROC curves,
confusion matrices, and feature importance cleanly separated from theoretical methodology.
"""

import streamlit as st
import pandas as pd
from src.evaluate import plot_confusion_matrices
from src.explainability import get_global_feature_importance
from src.interactive_plots import (
    plot_feature_importances_plotly,
    plot_metrics_comparison_plotly,
    plot_roc_curves_plotly,
)
from src.train import (
    load_all_evaluations,
    load_best_model,
    load_model_metadata,
    save_trained_artifacts,
    train_and_compare_models,
)
from src.ui_theme import apply_custom_theme, render_hero, render_theme_toggle

st.set_page_config(page_title="Model Benchmarks | Telco Churn", page_icon="⚙️", layout="wide")
render_theme_toggle()
apply_custom_theme()

render_hero(
    title="Model Performance & Comparative Evaluation",
    subtitle="Rigorous benchmark of classification pipelines on held-out test data (20% partition) across Precision, Recall, F1, and ROC-AUC.",
    badge_text="Module 3: Benchmark Studio",
)


def render_performance_view():
    all_evals = load_all_evaluations()
    metadata = load_model_metadata()
    best_pipeline = load_best_model()

    if not all_evals or not metadata or best_pipeline is None:
        st.warning("⚠️ No trained models detected. Click below to initialize the training workflow.")
        if st.button("🚀 Train & Benchmark Models (~10 seconds)"):
            with st.spinner("Training classification pipelines with 5-fold cross-validation..."):
                best_name, best_pipe, evals, meta = train_and_compare_models()
                save_trained_artifacts(best_pipe, meta, evals)
                st.success("Training complete! Refreshing page...")
                st.rerun()
        return

    tab_studio, tab_details = st.tabs(["⚙️ Interactive Benchmark Studio", "📖 Evaluation Methodology & Selection Details"])

    # =========================================================================
    # TAB 1: INTERACTIVE BENCHMARK STUDIO (WORKING MODE)
    # =========================================================================
    with tab_studio:
        # Champion Summary Banner
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #16a34a; margin-bottom: 1.2rem;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #15803d;">
                    🏆 Active Champion Model: {metadata.get('model_name', 'Random Forest')}
                </div>
                <div style="color: #475569; font-size: 0.95rem; margin-top: 0.2rem;">
                    Test Recall: <b>{metadata['metrics']['recall']:.1%}</b> &nbsp;|&nbsp; 
                    F1-Score: <b>{metadata['metrics']['f1_score']:.3f}</b> &nbsp;|&nbsp; 
                    ROC-AUC: <b>{metadata['metrics']['roc_auc']:.3f}</b> &nbsp;|&nbsp; 
                    Test Accuracy: <b>{metadata['metrics']['accuracy']:.1%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Performance Comparison Table
        st.subheader("📊 Cross-Model Comparison Table")
        table_rows = []
        for item in all_evals:
            table_rows.append({
                "Model Architecture": item["model_name"],
                "Accuracy": f"{item['accuracy']:.3f}",
                "Precision": f"{item['precision']:.3f}",
                "Recall (Catch Rate)": f"{item['recall']:.3f}",
                "F1-Score": f"{item['f1_score']:.3f}",
                "ROC-AUC": f"{item['roc_auc']:.3f}" if item.get("roc_auc") else "N/A",
                "5-Fold CV Recall (Mean ± Std)": f"{item.get('cv_recall_mean', 0):.3f} ± {item.get('cv_recall_std', 0):.3f}",
                "5-Fold CV F1 (Mean ± Std)": f"{item.get('cv_f1_mean', 0):.3f} ± {item.get('cv_f1_std', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.write("")

        # 2. Interactive Charts (Plotly)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_metrics = plot_metrics_comparison_plotly(all_evals)
            st.plotly_chart(fig_metrics, use_container_width=True)

        with col_c2:
            fig_roc = plot_roc_curves_plotly(all_evals)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()

        # 3. Diagnostic Confusion Matrices
        st.subheader("🎯 Test Set Confusion Matrices (n = 1,409)")
        fig_cm = plot_confusion_matrices(all_evals)
        st.pyplot(fig_cm, use_container_width=True)
        st.caption("Distribution of True Negatives, False Positives (Type I), False Negatives (Type II), and True Positives.")

        st.divider()

        # 4. Feature Importance
        st.subheader("🧠 Top Predictive Feature Importance")
        try:
            imp_df = get_global_feature_importance(best_pipeline, top_n=12)
            fig_imp = plot_feature_importances_plotly(imp_df, top_n=12)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance display not available: {str(e)}")

        st.divider()

        # On-Demand Retrain Trigger
        st.subheader("🔄 Retrain Pipelines")
        st.markdown("Re-run cross-validation and hyperparameter calibration on the latest data.")
        if st.button("⚡ Re-execute Full Training & Calibration Workflow"):
            with st.spinner("Retraining candidate pipelines on training partition..."):
                best_name, best_pipe, evals, meta = train_and_compare_models()
                save_trained_artifacts(best_pipe, meta, evals)
                st.success(f"Retraining complete! Champion: {best_name}")
                st.rerun()

    # =========================================================================
    # TAB 2: METHODOLOGY & SELECTION DETAILS (DOCUMENTATION MODE)
    # =========================================================================
    with tab_details:
        st.subheader("📖 Machine Learning Evaluation Methodology")

        st.markdown(
            f"""
            <div class="doc-box">
                <div class="doc-title">Why Random Forest Was Selected Over Nominal Accuracy</div>
                <div class="doc-text">
                    • <b>The Accuracy Paradox:</b> Gradient Boosting achieved 81.1% Accuracy, but its Recall was only 53.2%—it missed almost half of all departing subscribers.<br>
                    • <b>Business Cost Asymmetry:</b> In enterprise telecom, a <i>False Negative</i> costs $800 to $1,500+ in lost recurring revenue. A <i>False Positive</i> costs only a courtesy email or loyalty discount.<br>
                    • <b>Selection Score:</b> Random Forest achieved <b>80.2% Recall</b> and <b>0.628 F1-Score</b>, capturing 4 out of 5 churners while maintaining an 0.843 ROC-AUC.
                </div>
            </div>

            <div class="doc-box">
                <div class="doc-title">Hyperparameter Configuration & Validation Integrity</div>
                <div class="doc-text">
                    • <b>Partitions:</b> 80% Stratified Training (n=5,634) and 20% Held-Out Testing (n=1,409).<br>
                    • <b>Cross-Validation:</b> 5-Fold Stratified K-Fold CV applied strictly within the training set.<br>
                    • <b>Class Weighting:</b> Balanced class weighting applied to decision trees and random forest to handle the 73.5% / 26.5% class imbalance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Detailed Pipeline Hyperparameters")
        st.json(metadata.get("model_params", {}))


render_performance_view()
