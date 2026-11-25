"""
Script to generate the complete beginner-friendly Jupyter Notebook.
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
# 1. TITLE & INTRODUCTION
# ==========================================
add_md("""# 🏦 Bank-Term Deposit Subscription Prediction
## 📘 Complete Beginner's Guide to Machine Learning

Welcome to this beginner-friendly Machine Learning project! 👋 

If you are new to Data Science, Machine Learning, or Banking Analytics, **this notebook was designed specifically for you**. We will break down every single step with simple analogies, visual charts, and clearly commented code.

---

### ❓ What is the Problem We Are Solving?
Imagine you are the Head of Marketing at a retail bank. 
- The bank wants customers to open a **Term Deposit** (putting their savings into the bank for a fixed period to earn interest).
- The bank has a list of **45,000+ customers** to call on the phone.
- **The problem:** Calling 45,000 people takes thousands of hours, costs a lot of money, and annoys customers who aren't interested.
- In fact, historically, only **~11.7%** of people say "Yes"! Over **88%** say "No".

### 🎯 Our Mission:
Build a smart computer program (a **Machine Learning Classification Model**) that looks at a customer's profile (age, job, balance, loan history, etc.) and predicts:
> **"Will this customer subscribe to a term deposit? (Yes or No?)"**

With this model, bank staff can call **only the high-potential customers**, saving time, reducing costs, and boosting sales!
""")

# ==========================================
# 2. STEP 1: IMPORTING LIBRARIES & LOADING DATA
# ==========================================
add_md("""---
## 📦 Step 1: Importing Our Tools & Loading the Data

In Python, we use open-source "libraries" (toolkits) to do our work:
- **`pandas`**: Like Microsoft Excel inside Python. Used to store and view tables of data (DataFrames).
- **`numpy`**: Used for fast math and numbers.
- **`matplotlib` & `seaborn`**: Used to draw beautiful charts and graphs.
- **`scikit-learn`**: The world's most popular library for machine learning algorithms.
""")

add_code("""# Import standard Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make charts look clean and readable
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline

print("All libraries imported successfully!")
""")

add_md("""### 📂 Loading the Dataset
Our dataset is stored in `bank-full.csv`. In this file, columns are separated by semicolons (`;`). Let's load it and see how many rows and columns we have.
""")

add_code("""# Load dataset from the project root folder
df = pd.read_csv('../bank-full.csv', sep=';')

# Check dimensions (rows and columns)
print(f"Total Rows (Customers): {df.shape[0]:,}")
print(f"Total Columns (Features): {df.shape[1]}")

# Display the first 5 rows
df.head()
""")

add_md("""### 📋 Understanding the Columns (Features)
Here is what each column represents in plain English:

1. **Client Demographics**:
   - `age`: Customer's age (e.g. 35, 58)
   - `job`: Type of job (e.g. `management`, `technician`, `blue-collar`, `retired`, `student`)
   - `marital`: Marital status (`married`, `single`, `divorced`)
   - `education`: Education level (`primary`, `secondary`, `tertiary`, `unknown`)
2. **Financial Status**:
   - `default`: Does the customer already have credit in default? (`yes` / `no`)
   - `balance`: Average yearly bank balance in Euros (€)
   - `housing`: Does the customer have a housing loan / mortgage? (`yes` / `no`)
   - `loan`: Does the customer have a personal loan? (`yes` / `no`)
3. **Outreach & Communication History**:
   - `contact`: Contact communication type (`cellular`, `telephone`, `unknown`)
   - `month`: Last contact month of year (`jan` to `dec`)
   - `day`: Last contact day of the month ($1$ to $31$)
   - `duration`: Last contact duration in seconds (how long the phone call lasted)
   - `campaign`: Number of times the customer was contacted during this campaign
   - `pdays`: Days passed after the client was last contacted from a previous campaign ($-1$ means never contacted before)
   - `previous`: Number of contacts performed before this campaign
   - `poutcome`: Outcome of the previous marketing campaign (`success`, `failure`, `other`, `unknown`)
4. **Target (What we want to predict!)**:
   - `y`: Did the client subscribe to a term deposit? (`yes` or `no`)
""")

# ==========================================
# 3. STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
add_md("""---
## 🔍 Step 2: Exploratory Data Analysis (EDA) - Getting to Know the Data

Before we build any machine learning models, a good data scientist always explores the data first. 
Let's ask 4 key questions:
1. Is there any missing data?
2. How severe is the class imbalance (Yes vs No)?
3. Which customer jobs convert the most?
4. How does debt (loans) affect a customer's willingness to save?
""")

add_code("""# Check for missing values (NaN / Nulls)
print("--- Missing Values Check ---")
print(df.isnull().sum())
print("\\nGreat news! There are zero missing values in this dataset.")
""")

add_md("""### ⚖️ The Imbalance Problem: Target Distribution (`y`)
Let's see what percentage of people said "yes" vs "no".
""")

add_code("""# Count the target classes
counts = df['y'].value_counts()
percentages = df['y'].value_counts(normalize=True) * 100

print(f"No (Did not subscribe): {counts['no']:,} ({percentages['no']:.2f}%)")
print(f"Yes (Subscribed):        {counts['yes']:,} ({percentages['yes']:.2f}%)")

# Plot target distribution
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['No (88.3%)', 'Yes (11.7%)'], counts.values, color=['#3B82F6', '#10B981'], width=0.5)
ax.set_title('Term Deposit Subscription: Imbalanced Target (y)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Clients')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f'{yval:,}', ha='center', va='bottom', fontweight='bold')
plt.show()
""")

add_md("""💡 **Beginner's Lesson on Class Imbalance**:
Notice that **88.3%** of people said "No" and only **11.7%** said "Yes".
- If a lazy computer model simply predicted *"Everyone will say No"*, it would be **88.3% accurate**!
- But it would find **ZERO** subscribers and generate **ZERO** revenue for the bank.
- That is why in imbalanced problems, **Accuracy is a trap!** We must evaluate our models using **Precision, Recall, F1-Score, and ROC-AUC**.
""")

add_md("""### 👔 Question: Which jobs convert at the highest rate?""")

add_code("""# Calculate conversion rate by job
df['target_num'] = (df['y'] == 'yes').astype(int)
job_conv = df.groupby('job')['target_num'].agg(['count', 'mean']).sort_values('mean', ascending=False)
job_conv['Conversion %'] = job_conv['mean'] * 100

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=job_conv['Conversion %'], y=job_conv.index, palette='viridis', ax=ax)
ax.set_title('Term Deposit Conversion Rate by Job (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Conversion Rate (%)')
ax.axvline(x=11.7, color='red', linestyle='--', label='Average Base Rate (11.7%)')
ax.legend()
plt.tight_layout()
plt.show()
""")

add_md("""💡 **Key Insight**:
- **Students (28.7%)** and **Retirees (22.8%)** have the highest conversion rates! (More than **2x the average**).
- **Blue-collar workers (7.3%)** have the lowest conversion rate.
- Why? Retirees have lifetime savings and want safe, guaranteed interest. Students are eager for simple savings plans. Blue-collar workers often face tight monthly budgets with little spare liquidity.
""")

add_md("""### 💳 Question: What is the effect of Loans & Debt?""")

add_code("""# Conversion rate by Housing and Personal Loan
loan_conv = df.groupby(['housing', 'loan'])['target_num'].agg(['count', 'mean']).reset_index()
loan_conv['Conversion %'] = loan_conv['mean'] * 100
loan_conv['Profile'] = loan_conv.apply(lambda r: f"Housing: {r['housing']} | Loan: {r['loan']}", axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=loan_conv, x='Profile', y='Conversion %', palette='Blues_r', ax=ax)
ax.set_title('Impact of Existing Debt on Term Deposit Conversion', fontsize=14, fontweight='bold')
ax.set_ylabel('Conversion Rate (%)')
for i, v in enumerate(loan_conv['Conversion %']):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
""")

add_md("""💡 **Key Insight**:
- Customers with **NO housing loan and NO personal loan** convert at **18.2%**!
- Customers burdened with **both loans** convert at only **6.1%**!
- That is a **3x difference**! When people have heavy debt repayments, they cannot afford to lock money away in a term deposit.
""")

# ==========================================
# 4. STEP 3: PREPROCESSING & FEATURE ENGINEERING
# ==========================================
add_md("""---
## 🧹 Step 3: Preparing the Data for Machine Learning

Computers cannot read English words like `"married"` or `"secondary"` directly—they only understand numbers ($0, 1, 2, 3$).

In this step, we do 3 important tasks:
1. **Ordinal Encoding**: Converting ordered text into ranked numbers.  
   `primary` $\\to 1$, `secondary` $\\to 2$, `tertiary` $\\to 3$, `unknown` $\\to 0$.
2. **One-Hot Encoding**: Converting un-ordered categories into columns of 0s and 1s.  
   For example, `marital_single`, `marital_married`, `marital_divorced`.
3. **Feature Engineering**: Creating new, high-signal columns using our business logic:
   - `duration_per_campaign`: Call duration divided by number of calls. (Did the customer actually have a meaningful conversation?)
   - `is_debt_free`: 1 if the customer has no loans, 0 if they have debt.
   - `balance_to_age`: Bank balance divided by age. (How wealthy are they for their age?)
4. **Train / Test Split**: We split the data into **80% Training data** (the textbook the model learns from) and **20% Test data** (the final exam to check its accuracy on new customers).
""")

add_code("""# Import modular preprocessing pipeline from our src directory
import sys
sys.path.append('..')
from src.data_loader import get_train_test_data
from src.pipeline import build_preprocessor, get_feature_names

# 1. Stratified 80/20 Split (ensures equal proportion of Yes/No in train and test)
X_train, X_test, y_train, y_test = get_train_test_data('../bank-full.csv', test_size=0.20, random_state=42)

print(f"Training customers: {len(X_train):,}")
print(f"Testing customers:  {len(X_test):,}")

# 2. Build and fit our leak-free Scikit-Learn Preprocessing Pipeline
preprocessor = build_preprocessor()

# IMPORTANT: We FIT only on X_train (learning from training data), then transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = get_feature_names(preprocessor)
print(f"\\nProcessed Features Shape: {X_train_processed.shape}")
print(f"Total Engineered & Encoded Features: {len(feature_names)}")
print("First 10 feature names:", feature_names[:10])
""")

# ==========================================
# 5. STEP 4: MODEL TRAINING & BENCHMARKING
# ==========================================
add_md("""---
## 🤖 Step 4: Training Machine Learning Models (From Baseline to Champion)

We will compare several different types of machine learning algorithms:

1. **Logistic Regression (Baseline)**: The simplest model. Like drawing a straight line dividing the customers who subscribe from those who don't.
2. **Gaussian Naive Bayes**: A probability-based model using Bayes' Theorem.
3. **Random Forest**: An ensemble of 200 Decision Trees. Each tree asks a series of yes/no questions, and they vote on the final answer.
4. **XGBoost & CatBoost (Gradient Boosting)**: Trees built sequentially where each new tree specifically focuses on correcting the mistakes of the previous trees!
5. **Stacking Classifier (Our Champion!)**: A "meta-model" that combines the predictions of XGBoost, CatBoost, Random Forest, and Gradient Boosting together using a Logistic Regression judge!
""")

add_code("""# Load our pre-computed benchmark leaderboards
cv_leaderboard = pd.read_csv('../artifacts/metrics/cv_leaderboard.csv')
test_leaderboard = pd.read_csv('../artifacts/metrics/test_leaderboard.csv')

print("=== 5-FOLD STRATIFIED CROSS-VALIDATION LEADERBOARD ===")
cv_leaderboard[['Model', 'CV ROC-AUC (Mean)', 'CV F1-Score', 'CV Recall', 'CV Precision']].head(8)
""")

add_md("""### 🎯 Holdout Test Set Performance (The Real-World Exam)
Let's see how all models performed on the 9,043 unseen test customers:
""")

add_code("""# Display Holdout Test Set Leaderboard
test_leaderboard[['Model', 'ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall', 'Precision', 'Calibrated F1-Score', 'Optimal Threshold']]
""")

add_md("""💡 **Beginner's Guide to Metrics**:
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
  - Measures how well the model ranks good leads above bad leads.
  - $0.50$ = Random coin flip.
  - $1.00$ = Perfect ranking.
  - Our Stacking Classifier scored **0.9359** (Exceptional performance!).
- **Recall (88.1%)**: Out of all people who were actually going to say "Yes", our model successfully found **88% of them**!
- **Precision (44.2% default $\\to$ 56.6% calibrated)**: When our model tells a sales rep to call a customer, more than half of those calls convert! (Compared to just 11.7% by random calling).
""")

# ==========================================
# 6. STEP 5: VISUAL DIAGNOSTICS
# ==========================================
add_md("""---
## 📊 Step 5: Visualizing Model Performance

Let's look at the charts comparing all models:
1. **Model Leaderboard Comparison Bar Chart**
2. **ROC and Precision-Recall Curves**
3. **Confusion Matrices**
""")

add_code("""# Display saved comparative diagnostic figures
from IPython.display import Image, display

print("--- 1. Model Leaderboard Comparison ---")
display(Image(filename='../artifacts/figures/leaderboard_comparison.png'))

print("--- 2. ROC and Precision-Recall Curves ---")
display(Image(filename='../artifacts/figures/roc_pr_curves_comparison.png'))

print("--- 3. Confusion Matrices (Predicted vs Actual) ---")
display(Image(filename='../artifacts/figures/confusion_matrices_comparison.png'))
""")

# ==========================================
# 7. STEP 6: EXPLAINABLE AI (SHAP)
# ==========================================
add_md("""---
## 🔍 Step 6: Explainable AI with SHAP (Opening the Black Box)

### ❓ What is SHAP?
Imagine a soccer team wins a match 3-1. 
- The striker scored 2 goals.
- The goalkeeper made 5 incredible saves.
- The defender made a mistake that led to 1 goal conceded.

**SHAP (Shapley Additive exPlanations)** is a mathematical method based on Nobel-Prize winning game theory. For every single customer prediction, SHAP calculates:
> **"Exactly how much did each feature push the customer toward saying YES or toward saying NO?"**

Let's see the global SHAP summary plot:
""")

add_code("""# Display SHAP Global Beeswarm & Importance Plots
print("--- SHAP Beeswarm Plot (Feature Impact and Direction) ---")
display(Image(filename='../artifacts/figures/shap_summary_beeswarm.png'))

print("--- SHAP Global Feature Importance ---")
display(Image(filename='../artifacts/figures/shap_feature_importance_bar.png'))
""")

add_md("""💡 **How to Read the Beeswarm Plot**:
- **Features at the top** have the biggest impact on the prediction.
- **Red dots** mean high values of the feature (e.g. long call duration, high balance).
- **Blue dots** mean low values of the feature (e.g. short call, zero balance).
- **Points to the right of 0** push the probability **UP** (towards YES).
- **Points to the left of 0** push the probability **DOWN** (towards NO).

### 🔍 Explaining an Individual Customer (SHAP Waterfall Plot)
Let's see why the model gave a 94% probability to a high-conversion prospect:
""")

add_code("""# Display Local Waterfall Explanations
print("--- High-Propensity Client Explanation (P = 0.94) ---")
display(Image(filename='../artifacts/figures/shap_waterfall_high_propensity.png'))

print("--- Low-Propensity Client Explanation (P = 0.02) ---")
display(Image(filename='../artifacts/figures/shap_waterfall_low_propensity.png'))
""")

# ==========================================
# 8. STEP 7: MAKING PREDICTIONS ON NEW CUSTOMERS
# ==========================================
add_md("""---
## 🎯 Step 7: Making Live Predictions on a Brand New Customer

Now that our model is trained, let's see how a bank employee would use it in practice!
Suppose a new customer walks into the branch:
- **Age**: 65
- **Job**: Retired
- **Balance**: €15,000
- **Housing Loan**: No
- **Personal Loan**: No
- **Call Duration**: 400 seconds (long, engaged call)
- **Previous Outcome**: Success

Let's pass this customer through our pipeline!
""")

add_code("""import joblib

# Load saved champion model and preprocessor
champion_model = joblib.load('../artifacts/models/champion_model.joblib')
preprocessor = joblib.load('../artifacts/models/preprocessor.joblib')

# Create a sample customer profile
new_customer = pd.DataFrame([{
    'age': 65,
    'job': 'retired',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 15000,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'oct',
    'duration': 400,
    'campaign': 1,
    'pdays': 90,
    'previous': 2,
    'poutcome': 'success'
}])

# Transform through preprocessing pipeline
new_customer_proc = preprocessor.transform(new_customer)

# Get prediction probability
prob = champion_model.predict_proba(new_customer_proc)[0, 1]

print("=" * 50)
print(f"📊 Predicted Subscription Probability: {prob:.1%}")
if prob >= 0.70:
    print("🎯 Recommendation: VIP PRIORITY! Call immediately.")
elif prob >= 0.40:
    print("📈 Recommendation: High Potential. Send email brochure and follow up.")
else:
    print("⏸️ Recommendation: Low Propensity. Deprioritize to save costs.")
print("=" * 50)
""")

# ==========================================
# 9. STEP 8: STREAMLIT APP & CONCLUSION
# ==========================================
add_md("""---
## 🌐 Step 8: Interactive Streamlit Dashboard

To make this model easy for any bank manager or sales rep to use without writing code, we built an interactive **Streamlit Web Application** (`app.py`).

### How to Launch the Web App:
Open your terminal in the project directory and run:
```bash
streamlit run app.py
```
This opens a clean web interface in your browser where you can:
- Slide age, balance, duration, and job to see the live probability needle move in real time!
- Click preset personas (e.g. *High Propensity Retiree* vs *Indebted Worker*).
- Upload a CSV file of 1,000 customers to rank them from highest to lowest conversion probability!
- Inspect local SHAP explanations for any customer.

---

## 🎓 Summary of Skills Learned

In this project, you mastered:
1. **Tabular Machine Learning**: Handling real-world banking datasets with mixed numerical and categorical types.
2. **Class Imbalance Handling**: Why accuracy is misleading and how to use ROC-AUC, Recall, and Precision.
3. **Feature Engineering**: Creating domain-specific ratios like `duration_per_campaign`, `balance_to_age`, and debt indicators.
4. **Scikit-Learn Pipelines**: Writing clean, leak-free code that fits transformers only on training folds.
5. **Modern Model Families**: Baseline Logistic Regression, Random Forest, Gradient Boosting (XGBoost & CatBoost), and Stacking Ensembles.
6. **Explainable AI (SHAP)**: Giving transparency to non-technical stakeholders on why the AI made a decision.
7. **End-to-End Deployment**: Packaging models into Streamlit dashboards, Docker containers, and cloud deployment on AWS.

🎉 **Congratulations on completing this project!**
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

print("Successfully generated notebooks/bank_term_deposit_beginner_guide.ipynb!")
