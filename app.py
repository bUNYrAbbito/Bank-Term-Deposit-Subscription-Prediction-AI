"""
Bank Term Deposit Subscription Prediction - Streamlit Decision Support Application.
Production-ready interactive dashboard featuring:
1. Live single-prospect scoring with dynamic decision thresholds
2. Persona simulator presets (High, Moderate, Low propensity)
3. Instant SHAP local feature attribution and next-best-action guidance
4. Batch lead scoring and tier ranking with CSV export
5. Multi-model leaderboard and diagnostic visualizations
6. Global explainability insights (Gini importance & SHAP beeswarm)
7. AWS production deployment architecture & instructions
"""

import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bank Term Deposit AI Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PATHS & CONSTANTS ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "artifacts" / "models"
FIGURES_DIR = BASE_DIR / "artifacts" / "figures"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"

# Custom CSS for executive financial styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .decision-badge-yes {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .decision-badge-no {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED MODEL & ARTIFACT LOADERS ---
@st.cache_resource
def load_pipeline_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    models = {
        "Stacking Classifier (Champion)": joblib.load(MODELS_DIR / "stacking_classifier.joblib"),
        "CatBoost": joblib.load(MODELS_DIR / "catboost.joblib"),
        "XGBoost": joblib.load(MODELS_DIR / "xgboost.joblib"),
        "Ensemble Blend (Voting)": joblib.load(MODELS_DIR / "ensemble_blend_voting.joblib"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
        "Gradient Boosting": joblib.load(MODELS_DIR / "gradient_boosting.joblib"),
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "Deep Learning (MLP)": joblib.load(MODELS_DIR / "deep_learning_mlp.joblib"),
    }

    try:
        explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
    except Exception:
        explainer = None

    return preprocessor, feature_names, models, explainer


@st.cache_data
def load_metrics_and_data():
    cv_lb = pd.read_csv(METRICS_DIR / "cv_leaderboard.csv")
    test_lb = pd.read_csv(METRICS_DIR / "test_leaderboard.csv")
    feat_imp = pd.read_csv(METRICS_DIR / "feature_importances_xgboost.csv")
    with open(METRICS_DIR / "summary_report.json") as f:
        meta = json.load(f)
    return cv_lb, test_lb, feat_imp, meta


preprocessor, feature_names, models, explainer = load_pipeline_artifacts()
cv_lb, test_lb, feat_imp, meta = load_metrics_and_data()


# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=64)
st.sidebar.title("Term Deposit AI Predictor")
st.sidebar.markdown("**Enterprise Conversion Intelligence**")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Select Active Model",
    list(models.keys()),
    index=0,
    help="Champion model is the Stacking Classifier ensembling XGBoost, CatBoost, Random Forest, and Gradient Boosting."
)
active_model = models[selected_model_name]

st.sidebar.markdown("### Operational Decision Threshold")
threshold = st.sidebar.slider(
    "Classification Cut-off",
    min_value=0.10,
    max_value=0.90,
    value=float(meta.get("champion_optimal_threshold", 0.70)),
    step=0.01,
    help="Calibrated threshold balancing Precision vs Recall to maximize marketing efficiency."
)
st.sidebar.caption(f"Default: 0.50 | Calibrated Optimal for F1: **{meta.get('champion_optimal_threshold', 0.70):.2f}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Persona Simulator Presets")
preset_choice = st.sidebar.radio(
    "Load Prospect Preset:",
    ["Custom Input", "High Propensity Retiree", "Affluent Tech Executive", "Low Propensity Indebted Worker", "Young Student Prospect"],
    index=0,
)

# Preset values
if preset_choice == "High Propensity Retiree":
    preset_data = {
        "age": 65, "job": "retired", "marital": "married", "education": "secondary",
        "default": "no", "balance": 12500, "housing": "no", "loan": "no",
        "contact": "cellular", "day": 15, "month": "oct", "duration": 480,
        "campaign": 1, "pdays": 90, "previous": 2, "poutcome": "success"
    }
elif preset_choice == "Affluent Tech Executive":
    preset_data = {
        "age": 42, "job": "management", "marital": "single", "education": "tertiary",
        "default": "no", "balance": 28400, "housing": "no", "loan": "no",
        "contact": "cellular", "day": 12, "month": "jun", "duration": 360,
        "campaign": 2, "pdays": 180, "previous": 3, "poutcome": "success"
    }
elif preset_choice == "Low Propensity Indebted Worker":
    preset_data = {
        "age": 36, "job": "blue-collar", "marital": "married", "education": "primary",
        "default": "no", "balance": -450, "housing": "yes", "loan": "yes",
        "contact": "unknown", "day": 20, "month": "may", "duration": 85,
        "campaign": 4, "pdays": -1, "previous": 0, "poutcome": "unknown"
    }
elif preset_choice == "Young Student Prospect":
    preset_data = {
        "age": 22, "job": "student", "marital": "single", "education": "secondary",
        "default": "no", "balance": 1850, "housing": "no", "loan": "no",
        "contact": "cellular", "day": 10, "month": "sep", "duration": 310,
        "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"
    }
else:
    preset_data = {
        "age": 39, "job": "technician", "marital": "married", "education": "secondary",
        "default": "no", "balance": 1500, "housing": "yes", "loan": "no",
        "contact": "cellular", "day": 18, "month": "jul", "duration": 220,
        "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown"
    }

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Change call duration or previous outcome to evaluate real-time what-if conversion elasticity.")


# --- MAIN INTERFACE TABS ---
st.markdown('<div class="main-header">🏦 Bank-Term Deposit Subscription AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Conversion Scoring, Explainable AI (SHAP), and Next-Best-Action Decision Support</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🎯 Live Prospect Scoring & SHAP",
    "📊 Batch Lead Prioritization",
    "🏆 Model Leaderboard & Diagnostics",
    "🔍 Global Explainability & Insights",
    "☁️ AWS Production Deployment",
])


# ==============================================================================
# TAB 1: LIVE PROSPECT SCORING & SHAP
# ==============================================================================
with tabs[0]:
    st.markdown("### 👤 Prospect Profile & Outreach Parameters")
    st.markdown("Input demographic, balance, and campaign outreach details below to receive a real-time conversion probability and explainable decision support.")

    with st.form("prospect_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🧑 Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=preset_data["age"])
            job = st.selectbox(
                "Job Occupation",
                ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
                index=["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"].index(preset_data["job"])
            )
            marital = st.selectbox(
                "Marital Status",
                ["married", "single", "divorced"],
                index=["married", "single", "divorced"].index(preset_data["marital"])
            )
            education = st.selectbox(
                "Education Level",
                ["primary", "secondary", "tertiary", "unknown"],
                index=["primary", "secondary", "tertiary", "unknown"].index(preset_data["education"])
            )

        with col2:
            st.markdown("##### 💰 Financial & Debt Profile")
            balance = st.number_input("Average Yearly Balance (€)", value=preset_data["balance"], step=500)
            default = st.selectbox("Credit Default History", ["no", "yes"], index=["no", "yes"].index(preset_data["default"]))
            housing = st.selectbox("Housing Loan", ["no", "yes"], index=["no", "yes"].index(preset_data["housing"]))
            loan = st.selectbox("Personal Loan", ["no", "yes"], index=["no", "yes"].index(preset_data["loan"]))

        with col3:
            st.markdown("##### 📞 Campaign Outreach Attributes")
            contact = st.selectbox("Communication Type", ["cellular", "telephone", "unknown"], index=["cellular", "telephone", "unknown"].index(preset_data["contact"]))
            month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(preset_data["month"]))
            day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=preset_data["day"])
            duration = st.number_input("Last Call Duration (seconds)", min_value=0, max_value=5000, value=preset_data["duration"], step=30)
            campaign = st.number_input("Contacts in Current Campaign", min_value=1, max_value=60, value=preset_data["campaign"])
            pdays = st.number_input("Days Since Prior Campaign (-1 if uncontacted)", min_value=-1, max_value=1000, value=preset_data["pdays"])
            previous = st.number_input("Contacts Before Current Campaign", min_value=0, max_value=50, value=preset_data["previous"])
            poutcome = st.selectbox("Prior Campaign Outcome", ["unknown", "failure", "other", "success"], index=["unknown", "failure", "other", "success"].index(preset_data["poutcome"]))

        submit_btn = st.form_submit_button("🚀 Compute Prediction & Generate Explainability", use_container_width=True)

    # Process Form
    input_dict = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
    }
    input_df = pd.DataFrame([input_dict])

    # Transform through pipeline
    input_proc = preprocessor.transform(input_df)

    # Predict
    if hasattr(active_model, "predict_proba"):
        proba = float(active_model.predict_proba(input_proc)[0, 1])
    else:
        df_score = active_model.decision_function(input_proc)[0]
        proba = float(1 / (1 + np.exp(-df_score)))

    decision = proba >= threshold

    st.markdown("---")
    st.markdown("### 📋 Predictive Intelligence Scorecard")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.metric(
            label="Subscription Likelihood",
            value=f"{proba:.1%}",
            delta=f"{'+' if proba >= 0.117 else ''}{(proba - 0.117):.1%} vs Base (11.7%)",
        )

    with res_col2:
        if decision:
            badge = '<div class="decision-badge-yes">✅ High Conversion Potential</div>'
            recommendation = "Target Immediately"
        else:
            badge = '<div class="decision-badge-no">❌ Low Propensity</div>'
            recommendation = "Deprioritize / Nurture"
        st.markdown(f"**Classification (Cut-off: {threshold:.2f})**")
        st.markdown(badge, unsafe_allow_html=True)

    with res_col3:
        if proba >= 0.70:
            tier = "Tier 1: VIP Priority"
            tier_color = "#10B981"
        elif proba >= 0.40:
            tier = "Tier 2: High Potential"
            tier_color = "#3B82F6"
        elif proba >= 0.20:
            tier = "Tier 3: Moderate Potential"
            tier_color = "#F59E0B"
        else:
            tier = "Tier 4: Low Conversion"
            tier_color = "#EF4444"
        st.metric(label="Targeting Priority Tier", value=tier)

    with res_col4:
        # Expected value calculation assuming €150 profit per conversion, €5 cost per call
        expected_profit = (proba * 150) - 5
        st.metric(label="Expected Economic Value", value=f"€{expected_profit:.2f} / lead")

    # Probability Gauge Visual
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Subscription Probability ({selected_model_name})", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': "%"},
            'bar': {'color': "#2563EB"},
            'steps': [
                {'range': [0, threshold * 100], 'color': "#F3F4F6"},
                {'range': [threshold * 100, 100], 'color': "#E0E7FF"},
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Next-Best-Action Guidance
    st.markdown("#### 🎯 Strategic Next-Best-Action Guidance")
    if proba >= 0.65:
        st.success("""
        **Action Plan: Immediate Direct Outreach (High Touch)**
        - **Channel**: Priority direct phone call from senior relationship manager.
        - **Pitch**: Emphasize capital preservation, preferential rates for high balances, and long-term liquidity flexibility.
        - **Timing**: Follow up within 48 hours while engagement indicators are elevated.
        """)
    elif proba >= 0.35:
        st.info("""
        **Action Plan: Multi-Channel Nurturing**
        - **Channel**: Personalized email brochure detailing deposit tiers followed by a scheduled advisory call.
        - **Pitch**: Focus on low-risk return and comparison against inflation. Offer tailored term maturities (6, 12, or 24 months).
        - **Timing**: Re-engage mid-week during optimal morning call windows.
        """)
    else:
        st.warning("""
        **Action Plan: Resource Preservation & Automated Low-Cost Nurturing**
        - **Channel**: Deprioritize telemarketing calls. Include deposit offer in digital banking app notifications or quarterly statement mailers.
        - **Reason**: Outreach costs exceed expected conversion utility. Protect client from campaign fatigue.
        """)

    # SHAP Local Explanation
    st.markdown("#### 🔍 SHAP Local Attribution (Why did the model assign this score?)")
    if explainer is not None:
        try:
            # Compute TreeExplainer attribution
            xgb_est = models["XGBoost"]
            shap_obj = explainer(input_proc)
            if len(shap_obj.shape) == 3:
                vals = shap_obj.values[0, :, 1]
            elif len(shap_obj.shape) == 2:
                vals = shap_obj.values[0]
            else:
                vals = shap_obj.values

            contrib_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": vals,
                "Absolute Impact": np.abs(vals),
            }).sort_values("Absolute Impact", ascending=False).head(12)

            contrib_df["Direction"] = np.where(contrib_df["SHAP Value"] > 0, "Increases Subscription", "Decreases Subscription")
            contrib_df["Color"] = np.where(contrib_df["SHAP Value"] > 0, "#10B981", "#EF4444")

            fig_shap_local = px.bar(
                contrib_df.sort_values("SHAP Value", ascending=True),
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Increases Subscription": "#10B981", "Decreases Subscription": "#EF4444"},
                title="Top 12 Features Driving This Individual Prediction (SHAP Contribution)",
            )
            fig_shap_local.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_shap_local, use_container_width=True)
        except Exception as e:
            st.warning(f"Local SHAP explanation could not be rendered: {e}")
    else:
        st.info("SHAP explainer available in offline batch mode.")


# ==============================================================================
# TAB 2: BATCH LEAD PRIORITIZATION
# ==============================================================================
with tabs[1]:
    st.markdown("### 📊 Bulk Lead Scoring & Marketing Prioritization")
    st.markdown("Upload a CSV batch of prospects to rank leads by conversion probability, assign marketing tiers, and maximize campaign ROI.")

    upload_file = st.file_uploader("Upload Prospect Batch CSV (columns matching bank marketing dataset)", type=["csv"])

    col_batch_btn1, col_batch_btn2 = st.columns([1, 4])
    with col_batch_btn1:
        use_sample = st.button("📂 Load 1,000 Sample Leads from Test Set", use_container_width=True)

    if upload_file is not None:
        batch_raw = pd.read_csv(upload_file, sep=None, engine="python")
        st.success(f"Uploaded batch with {len(batch_raw)} records.")
    elif use_sample or "batch_raw" in st.session_state:
        from src.data_loader import get_train_test_data
        _, X_test_sample, _, y_test_sample = get_train_test_data()
        sample_df = X_test_sample.copy().head(1000)
        sample_df["actual_y"] = y_test_sample.head(1000).values
        st.session_state["batch_raw"] = sample_df
        batch_raw = sample_df
        st.info("Loaded 1,000 holdout prospects with known ground truth for validation.")
    else:
        batch_raw = None

    if batch_raw is not None:
        # Preprocess and score
        drop_cols = ["y", "actual_y", "target"]
        X_batch = batch_raw.drop(columns=[c for c in drop_cols if c in batch_raw.columns])
        X_batch_proc = preprocessor.transform(X_batch)

        if hasattr(active_model, "predict_proba"):
            batch_probas = active_model.predict_proba(X_batch_proc)[:, 1]
        else:
            df_s = active_model.decision_function(X_batch_proc)
            batch_probas = 1 / (1 + np.exp(-df_s))

        scored_df = batch_raw.copy()
        scored_df["Conversion_Probability"] = np.round(batch_probas, 4)
        scored_df["Predict_Subscribe"] = np.where(batch_probas >= threshold, "Yes", "No")
        scored_df["Priority_Tier"] = pd.cut(
            batch_probas,
            bins=[-0.01, 0.20, 0.40, 0.70, 1.0],
            labels=["Tier 4: Deprioritize", "Tier 3: Moderate", "Tier 2: High Potential", "Tier 1: VIP Target"]
        )

        # High level KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Prospects Scored", f"{len(scored_df):,}")
        k2.metric("Recommended for Outreach", f"{(scored_df['Predict_Subscribe'] == 'Yes').sum():,} ({(scored_df['Predict_Subscribe'] == 'Yes').mean():.1%})")
        k3.metric("Top Tier 1 & 2 Leads", f"{scored_df['Priority_Tier'].isin(['Tier 1: VIP Target', 'Tier 2: High Potential']).sum():,}")
        k4.metric("Avg Predicted Conversion", f"{scored_df['Conversion_Probability'].mean():.1%}")

        # Distribution Chart
        fig_hist = px.histogram(
            scored_df,
            x="Conversion_Probability",
            color="Priority_Tier",
            nbins=40,
            title="Distribution of Prospect Conversion Probabilities in Batch",
            color_discrete_map={
                "Tier 1: VIP Target": "#10B981",
                "Tier 2: High Potential": "#3B82F6",
                "Tier 3: Moderate": "#F59E0B",
                "Tier 4: Deprioritize": "#EF4444",
            }
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text=f"Cut-off ({threshold:.2f})")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🔝 Top Ranked Prospects for Immediate Outreach")
        ranked_display = scored_df.sort_values("Conversion_Probability", ascending=False)
        st.dataframe(ranked_display[["age", "job", "balance", "housing", "loan", "duration", "poutcome", "Conversion_Probability", "Priority_Tier", "Predict_Subscribe"]].head(25), use_container_width=True)

        # Download button
        csv_data = ranked_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prioritized Lead Roster (CSV)",
            data=csv_data,
            file_name="prioritized_term_deposit_leads.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================================================================
# TAB 3: MODEL LEADERBOARD & DIAGNOSTICS
# ==============================================================================
with tabs[2]:
    st.markdown("### 🏆 Comprehensive Model Leaderboard & Diagnostics")
    st.markdown("Rigorous benchmark of 10 candidate architectures evaluated under **5-Fold Stratified Cross-Validation** and a completely independent **20% Holdout Test Set** (9,043 clients).")

    # Metrics Summary Cards
    st.markdown("#### 🌟 Champion Architecture Highlights")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Champion Model", meta.get("champion_model", "Stacking Classifier"))
    m2.metric("Test ROC-AUC", f"{meta.get('champion_test_roc_auc', 0):.4f}")
    m3.metric("Test Default F1", f"{meta.get('champion_test_f1', 0):.4f}")
    m4.metric("Calibrated F1-Score", f"{meta.get('champion_calibrated_f1', 0):.4f}", f"Thresh: {meta.get('champion_optimal_threshold', 0):.2f}")

    col_lb1, col_lb2 = st.columns(2)
    with col_lb1:
        st.markdown("##### 🧪 5-Fold Stratified Cross-Validation Results")
        st.dataframe(cv_lb.style.format({
            "CV ROC-AUC (Mean)": "{:.4f}",
            "CV ROC-AUC (Std)": "{:.4f}",
            "CV F1-Score": "{:.4f}",
            "CV Recall": "{:.4f}",
            "CV Precision": "{:.4f}",
            "CV Accuracy": "{:.4f}",
            "CV PR-AUC": "{:.4f}",
            "Training Time (s)": "{:.1f}",
        }), use_container_width=True)

    with col_lb2:
        st.markdown("##### 🎯 Holdout Test Set Performance (N=9,043)")
        st.dataframe(test_lb.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "Optimal Threshold": "{:.2f}",
            "Calibrated F1-Score": "{:.4f}",
            "Calibrated Recall": "{:.4f}",
            "Calibrated Precision": "{:.4f}",
        }), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Model Comparison Charts")
    if (FIGURES_DIR / "leaderboard_comparison.png").exists():
        st.image(str(FIGURES_DIR / "leaderboard_comparison.png"), caption="Multi-Metric Model Comparison across ROC-AUC, F1-Score, Recall, and Precision", use_container_width=True)

    st.markdown("#### 📉 ROC Curves & Precision-Recall Curves")
    if (FIGURES_DIR / "roc_pr_curves_comparison.png").exists():
        st.image(str(FIGURES_DIR / "roc_pr_curves_comparison.png"), caption="Comparative ROC & Precision-Recall Curves across all 10 evaluated models", use_container_width=True)

    st.markdown("#### 🧩 Multi-Model Confusion Matrices")
    if (FIGURES_DIR / "confusion_matrices_comparison.png").exists():
        st.image(str(FIGURES_DIR / "confusion_matrices_comparison.png"), caption="Normalized and Absolute Confusion Matrices for all candidate models", use_container_width=True)


# ==============================================================================
# TAB 4: GLOBAL EXPLAINABILITY & INSIGHTS
# ==============================================================================
with tabs[3]:
    st.markdown("### 🔍 Global Explainability & SHAP Interpretability")
    st.markdown("Uncovering the underlying behavioral, financial, and outreach mechanisms driving client term deposit subscription decisions.")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("#### 🌟 Top 20 Global Feature Importances (Gain)")
        if (FIGURES_DIR / "feature_importance_top20.png").exists():
            st.image(str(FIGURES_DIR / "feature_importance_top20.png"), caption="XGBoost Top 20 Feature Importances", use_container_width=True)

    with col_exp2:
        st.markdown("#### 🐝 SHAP Summary Beeswarm Plot")
        if (FIGURES_DIR / "shap_summary_beeswarm.png").exists():
            st.image(str(FIGURES_DIR / "shap_summary_beeswarm.png"), caption="SHAP Values: Directionality & Impact on Subscription Propensity", use_container_width=True)

    st.markdown("---")
    col_water1, col_water2 = st.columns(2)
    with col_water1:
        st.markdown("#### 🚀 High-Propensity Client Waterfall")
        if (FIGURES_DIR / "shap_waterfall_high_propensity.png").exists():
            st.image(str(FIGURES_DIR / "shap_waterfall_high_propensity.png"), caption="SHAP Waterfall for High-Conversion Client", use_container_width=True)

    with col_water2:
        st.markdown("#### 📉 Low-Propensity Client Waterfall")
        if (FIGURES_DIR / "shap_waterfall_low_propensity.png").exists():
            st.image(str(FIGURES_DIR / "shap_waterfall_low_propensity.png"), caption="SHAP Waterfall for Low-Conversion Client", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💡 Key Business Takeaways from SHAP Analysis")
    st.markdown("""
    1. **Call Engagement (`duration` & `duration_per_campaign`)**: By far the strongest operational signal. A call lasting > 5 minutes indicates high client receptivity.
    2. **Previous Success (`poutcome_success`)**: Historical campaign success produces the highest positive SHAP bump (+2.1 log-odds), translating to a **64.7% conversion rate**.
    3. **Financial Burden & Debt (`housing_yes`, `loan_yes`, `total_loans`)**: Clients with mortgage or personal loans demonstrate significantly depressed conversion likelihood. Freeing capital for deposits requires debt-free or low-debt clients.
    4. **Client Life Stage & Wealth (`job_retired`, `job_student`, `balance_to_age`)**: Retirees and students have the highest conversion rates (23-29%). Retirees seek low-risk fixed income for retirement wealth; students are receptive to automated savings programs.
    5. **Outreach Timing (`month_sin`, `month_cos`, `day`)**: March, September, and October exhibit disproportionately higher conversion rates compared to saturated marketing months like May.
    """)


# ==============================================================================
# TAB 5: AWS PRODUCTION DEPLOYMENT
# ==============================================================================
with tabs[4]:
    st.markdown("### ☁️ AWS Cloud Production Deployment Architecture")
    st.markdown("Architectural blueprint and step-by-step instructions for deploying this predictive application on Amazon Web Services (AWS).")

    st.markdown("""
    ```mermaid
    flowchart LR
        User[Banking Marketing Team] --> Route53[Amazon Route 53 / DNS]
        Route53 --> ALB[Application Load Balancer]
        ALB --> ECS[Amazon ECS / Fargate Cluster]
        subgraph AWS Cloud
            ECS --> Docker[Streamlit Docker Container: Port 8501]
            Docker --> ModelPipeline[Scikit-Learn & Stacking Pipeline]
            ModelPipeline --> S3[(Amazon S3: Model Artifacts)]
            Docker --> CloudWatch[Amazon CloudWatch: Monitoring & Logs]
        end
    ```
    """)

    col_dep1, col_dep2 = st.columns(2)

    with col_dep1:
        st.markdown("#### 🐳 Option 1: Containerized Deployment via AWS ECS / Fargate")
        st.markdown("""
        **Step 1: Build Docker Container locally**
        ```bash
        docker build -t bank-deposit-predictor:latest .
        ```

        **Step 2: Authenticate with Amazon Elastic Container Registry (ECR)**
        ```bash
        aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
        ```

        **Step 3: Tag and Push Image to ECR**
        ```bash
        docker tag bank-deposit-predictor:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/bank-deposit-predictor:latest
        docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/bank-deposit-predictor:latest
        ```

        **Step 4: Create ECS Task Definition & Fargate Service**
        - Launch Amazon ECS cluster with Fargate launch type (serverless).
        - Configure Container Port `8501`.
        - Attach Application Load Balancer (ALB) for HTTPS SSL termination and auto-scaling.
        """)

    with col_dep2:
        st.markdown("#### 🚀 Option 2: Serverless AWS App Runner")
        st.markdown("""
        AWS App Runner is the fastest managed container service for Streamlit applications:
        1. Open **AWS App Runner Console** -> *Create Service*.
        2. Select **Container Registry** -> Choose your ECR image.
        3. Set Port to `8501`.
        4. Configure 1 vCPU and 2 GB Memory.
        5. Enable Auto-Scaling (1 to 5 instances based on concurrent requests).
        6. App Runner provides an automated HTTPS URL with zero infrastructure management!

        #### ⚡ Option 3: AWS Elastic Beanstalk
        ```bash
        eb init -p docker bank-term-deposit-app --region us-east-1
        eb create bank-prod-env --instance_type t3.medium
        eb open
        ```
        """)

    st.markdown("---")
    st.markdown("#### 🛡️ Production Best Practices & MLOps Considerations")
    st.markdown("""
    - **Data Drift & Concept Drift Monitoring**: Schedule weekly Amazon CloudWatch Alarms comparing the distribution of incoming prospect features against the baseline training distribution.
    - **Artifact Versioning**: Store retrained `.joblib` model binaries in an Amazon S3 bucket with versioning and SHA256 checksums.
    - **Role-Based Access Control (RBAC)**: Protect sensitive customer financial data using AWS IAM roles with least-privilege policies.
    - **Automated CI/CD**: Integrate GitHub Actions to trigger automated unit testing, model evaluation, and deployment to ECR on pull requests.
    """)
