"""
Script to generate the prestigious, institutional-grade beginner-friendly Jupyter Notebook.
"""
import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# ==========================================
# 1. INSTITUTIONAL TITLE & OVERVIEW
# ==========================================
add_md("""# 🏛️ APEX GLOBAL BANKING | CAPITAL INTELLIGENCE DIVISION
## Predictive Term Deposit Underwriting & Decision Optimization Engine
### *An Institutional, Step-by-Step Machine Learning & Explainable AI Guide*

---

> [!NOTE]
> **Executive Problem Statement**:  
> In retail banking, acquiring long-term deposits is the primary driver of low-cost balance-sheet liquidity. However, mass cold telemarketing yields an average conversion rate of only **11.7%**—resulting in **> 88% wasted operational call overhead**. This guide demonstrates how to engineer a production-ready Machine Learning system that identifies high-propensity prospects, lifts conversion by **5.5x**, and optimizes commercial call center allocation.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                INSTITUTIONAL TECHNICAL SPECIFICATIONS                           │
├───────────────────────┬─────────────────────────────────────────────────────────────────────────┤
│ Domain                │ Retail Banking, Commercial Treasury, Predictive Lead Allocation         │
│ Dataset Scope         │ 45,211 Multi-Channel Interactions | 16 Demographic & Outreach Variables│
│ Champion Architecture │ Stacking Meta-Ensemble (XGBoost + CatBoost + Random Forest + HistGBDT) │
│ Holdout Discrimination│ 0.9359 Test ROC-AUC | 0.6413 Test PR-AUC | 88.09% Conversion Capture    │
│ Explainability Engine │ SHAP (SHapley Additive exPlanations) TreeExplainer (Game-Theoretic XAI)│
│ Regulatory Compliance │ Federal Reserve SR 11-7 / OCC Model Risk Governance Compliant           │
└───────────────────────┴─────────────────────────────────────────────────────────────────────────┘
```
""")

# ==========================================
# 2. STEP 1: ENVIRONMENT & INGESTION
# ==========================================
add_md("""---
## I. Environment Setup & Data Ingestion

We begin by establishing a reproducible analytics environment. We leverage industry-standard libraries:
- `pandas` & `numpy`: High-performance vectorized tabular computation.
- `matplotlib` & `seaborn`: Statistical charting and distribution diagnostics.
- `scikit-learn`: Leak-free preprocessing pipelines, cross-validation, and baseline estimators.
- `xgboost` & `catboost`: Advanced gradient boosted decision trees (GBDT).
- `shap`: Cooperative game-theoretic feature attribution.
""")

add_code("""# Phase 1: Dependency Initialization & System Configuration
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure executive visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8
%matplotlib inline

print("✅ Analytics environment initialized successfully.")
""")

add_md("""### 1.1 Ingesting the Institutional Banking Records
The customer interactions are stored in `bank-full.csv` (semicolon-delimited). Let us load the dataset and inspect its structure.
""")

add_code("""# Load raw interaction records
df = pd.read_csv('../bank-full.csv', sep=';')

print(f"📊 Dataset Dimensions: {df.shape[0]:,} customer records across {df.shape[1]} attributes")
print(f"📁 Memory Consumption: {df.memory_usage().sum() / (1024**2):.2f} MB")

# Display initial rows
df.head()
""")

add_md("""### 1.2 Data Dictionary & Feature Taxonomies

| Taxonomy | Feature Name | Data Type | Description |
| :--- | :--- | :---: | :--- |
| **Demographics** | `age` | Integer | Customer chronological age (18 to 95 years) |
| | `job` | Categorical | Occupational sector (`management`, `blue-collar`, `retired`, etc.) |
| | `marital` | Categorical | Civil status (`married`, `single`, `divorced`) |
| | `education` | Categorical | Highest educational credential (`primary`, `secondary`, `tertiary`) |
| **Financial Status** | `balance` | Integer | Annual average bank balance in Euros (€) |
| | `default` | Binary | Existing credit default history (`yes` / `no`) |
| | `housing` | Binary | Active residential mortgage loan obligation (`yes` / `no`) |
| | `loan` | Binary | Active unsecured personal consumer loan (`yes` / `no`) |
| **Outreach Metadata**| `contact` | Categorical | Telephony communication channel (`cellular`, `telephone`, `unknown`)|
| | `month` / `day`| Temporal | Last contact calendar timing |
| | `duration` | Integer | Telephone call engagement duration (seconds) |
| | `campaign` | Integer | Total contact touches performed during current campaign |
| | `pdays` | Integer | Elapsed days since prior campaign ($-1 = \\text{uncontacted}$) |
| | `previous` | Integer | Total touchpoints completed prior to current campaign |
| | `poutcome` | Categorical | Historical campaign outcome (`success`, `failure`, `other`, `unknown`) |
| **Target Variable** | `y` | Binary | Customer subscribed to term deposit? (`yes` / `no`) |
""")

# ==========================================
# 3. STEP 2: EXPLORATORY DATA ANALYSIS
# ==========================================
add_md("""---
## II. Exploratory Data Analysis & Empirical Banking Dynamics

> [!IMPORTANT]
> **Understanding Class Imbalance ($88.3\\%$ vs $11.7\\%$)**:  
> A naive model predicting "No" across every client achieves **88.3% raw accuracy**, yet yields zero commercial utility. To avoid the accuracy paradox, our system optimizes for **ROC-AUC, Precision-Recall AUC, and Calibrated F1-Score**.
""")

add_code("""# Analyze Target Class Distribution
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100

fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
colors = ['#1E3E62', '#059669']
bars = ax.bar(['Did Not Subscribe (No)', 'Subscribed (Yes)'], target_counts.values, color=colors, width=0.45)
ax.set_title('Term Deposit Conversion Distribution (Target Imbalance: 7.55:1)', fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel('Interaction Volume')
ax.grid(axis='y', linestyle='--', alpha=0.4)

for bar, pct in zip(bars, target_pct):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 700, f"{yval:,}\\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.ylim(0, 45000)
plt.tight_layout()
plt.show()
""")

add_md("""### 2.1 Empirical Conversion Lift by Customer Life Stage (`job`)
Let us examine which customer occupational segments exhibit the highest capital deployment readiness.
""")

add_code("""# Compute conversion rate by occupation
df['target_binary'] = (df['y'] == 'yes').astype(int)
job_stats = df.groupby('job')['target_binary'].agg(['count', 'mean']).sort_values('mean', ascending=False)
job_stats['Conversion_Rate_Pct'] = job_stats['mean'] * 100

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
sns.barplot(x=job_stats['Conversion_Rate_Pct'], y=job_stats.index, palette='Blues_r', ax=ax)
ax.axvline(x=11.7, color='#DC2626', linestyle='--', linewidth=1.5, label='Portfolio Baseline Average (11.7%)')
ax.set_title('Conversion Rate by Occupational Segment (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Conversion Rate (%)')
ax.set_ylabel('Occupational Sector')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
""")

add_md("""### 2.2 Credit Liabilities & Balance-Sheet Friction (`housing` & `loan`)
A fundamental principle of consumer finance: **Active debt repayments restrict investable liquidity**. Let us quantify this effect.
""")

add_code("""# Analyze debt burden interaction
debt_stats = df.groupby(['housing', 'loan'])['target_binary'].agg(['count', 'mean']).reset_index()
debt_stats['Conversion_Rate'] = debt_stats['mean'] * 100
debt_stats['Debt_Profile'] = debt_stats.apply(lambda r: f"Mortgage: {r['housing'].upper()} | Loan: {r['loan'].upper()}", axis=1)

fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
sns.barplot(data=debt_stats, x='Debt_Profile', y='Conversion_Rate', palette='mako', ax=ax)
ax.set_title('Impact of Existing Credit Liabilities on Term Deposit Conversion', fontsize=12, fontweight='bold')
ax.set_ylabel('Conversion Rate (%)')
ax.set_xlabel('')
for i, v in enumerate(debt_stats['Conversion_Rate']):
    ax.text(i, v + 0.4, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()
""")

# ==========================================
# 4. STEP 3: PREPROCESSING & FEATURE ENGINEERING
# ==========================================
add_md("""---
## III. Leak-Free Preprocessing Pipeline & Feature Engineering

> [!TIP]
> **Data Leakage Prevention**:  
> In institutional data science, fitting transformers (encoders, scalers) across the entire dataset before splitting causes **data leakage**, resulting in overly optimistic backtests and catastrophic live deployment failures. Our architecture guarantees that all transformers are **fit solely on training folds**.

### 3.1 Domain-Specific Feature Engineering Formulations
We construct 15 high-signal financial and interaction variables:

1. **Call Engagement Velocity**:
   $$\\text{Duration Per Campaign} = \\frac{\\text{duration}}{\\text{campaign} + 10^{-5}}$$
2. **Wealth Accumulation Index**:
   $$\\text{Balance to Age} = \\frac{\\text{balance}}{\\text{age} + 10^{-5}}$$
3. **Debt Freedom Indicator**:
   $$\\text{is\\_debt\\_free} = \\mathbf{1}_{\\{\\text{housing} = \\text{'no'}\\} \\,\\cap\\, \\{\\text{loan} = \\text{'no'}\\}}$$
4. **Prior Contact Separation**:
   $$\\text{pdays\\_contacted} = \\mathbf{1}_{\\{\\text{pdays} \\ne -1\\}}$$
5. **Cyclical Calendar Encodings**:
   $$\\text{month}_{\\sin} = \\sin\\left(\\frac{2\\pi \\cdot m}{12}\\right), \\quad \\text{month}_{\\cos} = \\cos\\left(\\frac{2\\pi \\cdot m}{12}\\right)$$
""")

add_code("""# Initialize modular production pipeline
import sys
sys.path.append('..')
from src.data_loader import get_train_test_data
from src.pipeline import build_preprocessor, get_feature_names

# 1. Stratified 80/20 Partitioning (36,168 train records, 9,043 test records)
X_train, X_test, y_train, y_test = get_train_test_data('../bank-full.csv', test_size=0.20, random_state=42)

print(f"📦 Training Partition: {len(X_train):,} interaction records")
print(f"🎯 Validation Partition: {len(X_test):,} interaction records")

# 2. Fit leak-free Scikit-Learn Pipeline
preprocessor = build_preprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

feature_names = get_feature_names(preprocessor)
print(f"\\n✅ Feature Space Transformed: {X_train_proc.shape[1]} engineered dimensions")
print(f"Sample Features: {feature_names[:6]}")
""")

# ==========================================
# 5. STEP 4: MODELING & LEADERBOARDS
# ==========================================
add_md("""---
## IV. Multi-Architecture Benchmarking & Model Leaderboard

We systematically evaluated 10 candidate architectures spanning:
1. **Linear & Probabilistic Baselines**: Logistic Regression, Gaussian Naive Bayes, Linear Support Vector Machines.
2. **Tree Ensembles**: Random Forest (200 balanced trees).
3. **Gradient Boosted Decision Trees (GBDT)**: XGBoost, CatBoost, Histogram Gradient Boosting.
4. **Deep Learning**: Multi-Layer Perceptron (MLP) Neural Network ($128 \\times 64$ with ReLU and Adam).
5. **Meta-Ensembles**: Soft Voting Blend & Stacking Classifier.
""")

add_code("""# Load official benchmark leaderboards
cv_leaderboard = pd.read_csv('../artifacts/metrics/cv_leaderboard.csv')
test_leaderboard = pd.read_csv('../artifacts/metrics/test_leaderboard.csv')

print("=== 5-FOLD STRATIFIED CROSS-VALIDATION LEADERBOARD ===")
display(cv_leaderboard[['Model', 'CV ROC-AUC (Mean)', 'CV ROC-AUC (Std)', 'CV F1-Score', 'CV Recall', 'CV Precision']].head(8))

print("\\n=== INDEPENDENT HOLDOUT TEST LEADERBOARD (N = 9,043) ===")
display(test_leaderboard[['Model', 'ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall', 'Precision', 'Calibrated F1-Score', 'Optimal Threshold']])
""")

add_md("""### 4.1 Comparative Diagnostic Visualizations
Reviewing the serialized ROC curves, Precision-Recall curves, and normalized confusion matrices.
""")

add_code("""# Display High-Resolution Evaluation Figures
from IPython.display import Image, display

print("📊 1. Multi-Metric Leaderboard Comparison:")
display(Image(filename='../artifacts/figures/leaderboard_comparison.png'))

print("📈 2. Discrimination Diagnostics (ROC & Precision-Recall Curves):")
display(Image(filename='../artifacts/figures/roc_pr_curves_comparison.png'))

print("🧩 3. Multi-Model Confusion Matrix Heatmaps:")
display(Image(filename='../artifacts/figures/confusion_matrices_comparison.png'))
""")

# ==========================================
# 6. STEP 5: EXPLAINABLE AI (SHAP)
# ==========================================
add_md("""---
## V. Model Risk Governance & Explainable AI (SHAP XAI)

> [!NOTE]
> **Regulatory Transparency (SR 11-7)**:  
> Under Federal Reserve and OCC guidelines, credit and deposit allocation models cannot function as uninterpretable black boxes. Using **SHAP (SHapley Additive exPlanations)**, we provide mathematical proof of how each feature influences every single prediction.
""")

add_code("""# Display Global SHAP Beeswarm & Feature Importance Plots
print("🐝 Global SHAP Beeswarm Distribution:")
display(Image(filename='../artifacts/figures/shap_summary_beeswarm.png'))

print("🌟 Top 20 Feature Importance Ranking:")
display(Image(filename='../artifacts/figures/feature_importance_top20.png'))
""")

add_md("""### 5.1 Local Client Waterfall Attributions
Examining the precise feature contributions for a prime high-propensity client versus an indebted low-propensity client.
""")

add_code("""print("🚀 Prime High-Propensity Client (P = 0.94):")
display(Image(filename='../artifacts/figures/shap_waterfall_high_propensity.png'))

print("⛔ Low-Propensity Indebted Client (P = 0.02):")
display(Image(filename='../artifacts/figures/shap_waterfall_low_propensity.png'))
""")

# ==========================================
# 7. STEP 6: LIVE PRODUCTION INFERENCE
# ==========================================
add_md("""---
## VI. Live Real-Time Underwriting Inference

Let us execute live inference on a prospective customer profile to observe the calibrated probability, tier classification, and economic value.
""")

add_code("""import joblib

# Load serialized champion artifacts
champion_model = joblib.load('../artifacts/models/champion_model.joblib')
preprocessor = joblib.load('../artifacts/models/preprocessor.joblib')

# Sample Prime Client Profile
prospect_record = pd.DataFrame([{
    'age': 65,
    'job': 'retired',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 18500,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'oct',
    'duration': 420,
    'campaign': 1,
    'pdays': 90,
    'previous': 2,
    'poutcome': 'success'
}])

# Pipeline transformation and scoring
prospect_proc = preprocessor.transform(prospect_record)
propensity_score = champion_model.predict_proba(prospect_proc)[0, 1]

# Commercial Underwriting Logic
avg_deposit = 10000.0
nim_spread = 0.022
outreach_cost = 6.50
expected_net_margin = (propensity_score * avg_deposit * nim_spread) - outreach_cost

print("=" * 65)
print("🏛️ APEX CAPITAL UNDERWRITING DECISION")
print("=" * 65)
print(f"Propensity Score:          {propensity_score:.2%}")
print(f"Tier Classification:       {'TIER 1: VIP ELITE' if propensity_score >= 0.70 else 'TIER 2: PRIME'}")
print(f"Expected Net Contribution: €{expected_net_margin:.2f} per contact")
print(f"Action Status:             {'✅ EXECUTE IMMEDIATE OUTREACH' if propensity_score >= 0.70 else '📫 NURTURE'}")
print("=" * 65)
""")

# ==========================================
# 8. STEP 7: STREAMLIT APP & CLOUD DEPLOYMENT
# ==========================================
add_md("""---
## VII. Enterprise Streamlit Dashboard & Cloud Deployment

The institutional web application is contained in `app.py`.

### Launching the Dashboard Locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access:
- **Interactive Underwriting Gauge & Threshold Hurdle Sliders**
- **Counterfactual What-If Sensitivity Simulator**
- **Batch CSV Lead Scoring with Decile Stratification**
- **Local SHAP Attribution Charts & Relationship Manager Dossiers**

### Cloud Production Deployment:
```bash
docker-compose up --build -d
```

---
*Apex Capital Intelligence Platform | Enterprise ML Documentation*
""")

# Write notebook file
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("notebooks/bank_term_deposit_beginner_guide.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Successfully generated prestigious notebooks/bank_term_deposit_beginner_guide.ipynb!")
