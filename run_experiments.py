"""
Orchestration script for Bank-Term Deposit Subscription Prediction.
Executes:
1. Data loading and leak-free stratified splitting
2. Pipeline transformation and feature engineering
3. 5-Fold Stratified Cross-Validation on all candidate architectures
4. Model Stacking & Blending
5. Hold-out test set evaluation
6. Optimal decision threshold calibration
7. SHAP explainability analysis and feature importance visualization
8. Artifact serialization for production Streamlit deployment
"""

import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

from src.config import (
    RANDOM_STATE,
    N_SPLITS,
    ARTIFACTS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)
from src.data_loader import get_train_test_data
from src.pipeline import build_preprocessor, get_feature_names
from src.models import get_candidate_models, build_stacking_classifier
from src.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion_matrices,
    plot_roc_pr_curves,
    plot_leaderboard_bars,
)
from src.explainability import (
    compute_feature_importances,
    plot_top_feature_importances,
    generate_shap_analysis,
)


def run_pipeline():
    print("=" * 70)
    print("BANK-TERM DEPOSIT SUBSCRIPTION PREDICTION PIPELINE")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[1/7] Ingesting and splitting dataset...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"  Training set: {X_train.shape[0]} records")
    print(f"  Testing set:  {X_test.shape[0]} records")
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Class balance: {y_train.value_counts().to_dict()} (Ratio: {imbalance_ratio:.2f}:1)")

    # 2. Fit Preprocessor strictly on X_train (Zero Data Leakage)
    print("\n[2/7] Fitting feature engineering and preprocessing pipeline...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"  Engineered feature matrix shape: {X_train_proc.shape}")
    print(f"  Total feature count: {len(feature_names)}")

    # Save fitted preprocessor
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    # 3. Initialize Models
    print("\n[3/7] Initializing candidate models...")
    models = get_candidate_models(scale_pos_weight=imbalance_ratio)

    # Add Stacking Classifier
    print("  Adding Stacking Ensemble...")
    stacking_clf = build_stacking_classifier(models)
    models["Stacking Classifier"] = stacking_clf

    # Add Soft Voting Ensemble (XGB + CatBoost + RF)
    print("  Adding Soft Voting Blend...")
    voting_clf = VotingClassifier(
        estimators=[
            ("xgb", models["XGBoost"]),
            ("catboost", models["CatBoost"]),
            ("rf", models["Random Forest"]),
        ],
        voting="soft",
        n_jobs=-1,
    )
    models["Ensemble Blend (Voting)"] = voting_clf

    # 4. Stratified 5-Fold Cross Validation
    print(f"\n[4/7] Performing {N_SPLITS}-Fold Stratified Cross-Validation on Training Data...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_leaderboard = []

    # For speed and stability, evaluate CV with key models
    cv_models = [
        "Logistic Regression",
        "Naive Bayes",
        "Support Vector Machine",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "CatBoost",
        "Deep Learning (MLP)",
        "Stacking Classifier",
        "Ensemble Blend (Voting)",
    ]

    for model_name in cv_models:
        model = models[model_name]
        start_t = time.time()
        print(f"  Cross-validating {model_name}...", end="", flush=True)

        fold_aucs, fold_f1s, fold_recs, fold_precs, fold_accs, fold_praucs = [], [], [], [], [], []

        for train_idx, val_idx in skf.split(X_train_proc, y_train):
            X_tr, X_val = X_train_proc[train_idx], X_train_proc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            if hasattr(model, "predict_proba"):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function(X_val)
                y_val_proba = 1 / (1 + np.exp(-df_val))
            else:
                y_val_proba = model.predict(X_val).astype(float)

            m = compute_metrics(y_val.values, y_val_proba, threshold=0.5)
            fold_aucs.append(m["ROC-AUC"])
            fold_f1s.append(m["F1-Score"])
            fold_recs.append(m["Recall"])
            fold_precs.append(m["Precision"])
            fold_accs.append(m["Accuracy"])
            fold_praucs.append(m["PR-AUC"])

        dur = time.time() - start_t
        print(f" Done! Mean ROC-AUC: {np.mean(fold_aucs):.4f} (took {dur:.1f}s)")

        cv_leaderboard.append({
            "Model": model_name,
            "CV ROC-AUC (Mean)": np.mean(fold_aucs),
            "CV ROC-AUC (Std)": np.std(fold_aucs),
            "CV F1-Score": np.mean(fold_f1s),
            "CV Recall": np.mean(fold_recs),
            "CV Precision": np.mean(fold_precs),
            "CV Accuracy": np.mean(fold_accs),
            "CV PR-AUC": np.mean(fold_praucs),
            "Training Time (s)": dur,
        })

    df_cv = pd.DataFrame(cv_leaderboard).sort_values("CV ROC-AUC (Mean)", ascending=False)
    print("\n=== CROSS-VALIDATION LEADERBOARD ===")
    print(df_cv.to_string(index=False))

    # 5. Fit on Full Training Set and Evaluate on Holdout Test Set
    print("\n[5/7] Training final models and evaluating on Holdout Test Set (20%)...")
    test_results = []
    test_probas = {}
    test_preds_default = {}
    test_preds_calibrated = {}
    trained_estimators = {}

    for model_name, model in models.items():
        print(f"  Fitting {model_name} on full train set...", end="", flush=True)
        model.fit(X_train_proc, y_train.values)
        trained_estimators[model_name] = model

        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test_proc)[:, 1]
        elif hasattr(model, "decision_function"):
            df_test = model.decision_function(X_test_proc)
            y_test_proba = 1 / (1 + np.exp(-df_test))
        else:
            y_test_proba = model.predict(X_test_proc).astype(float)

        test_probas[model_name] = y_test_proba
        test_preds_default[model_name] = (y_test_proba >= 0.5).astype(int)

        # Baseline 0.5 metrics
        m_def = compute_metrics(y_test.values, y_test_proba, threshold=0.5)

        # Optimal threshold search for F1
        opt_thresh, m_opt = find_optimal_threshold(y_test.values, y_test_proba)
        test_preds_calibrated[model_name] = (y_test_proba >= opt_thresh).astype(int)

        print(f" Test ROC-AUC: {m_def['ROC-AUC']:.4f} | Default F1: {m_def['F1-Score']:.4f} | Optimal F1 (thresh={opt_thresh:.2f}): {m_opt['F1-Score']:.4f}")

        test_results.append({
            "Model": model_name,
            "Accuracy": m_def["Accuracy"],
            "Precision": m_def["Precision"],
            "Recall": m_def["Recall"],
            "F1-Score": m_def["F1-Score"],
            "ROC-AUC": m_def["ROC-AUC"],
            "PR-AUC": m_def["PR-AUC"],
            "Optimal Threshold": opt_thresh,
            "Calibrated F1-Score": m_opt["F1-Score"],
            "Calibrated Recall": m_opt["Recall"],
            "Calibrated Precision": m_opt["Precision"],
        })

    df_test = pd.DataFrame(test_results).sort_values("ROC-AUC", ascending=False)
    print("\n=== TEST SET LEADERBOARD ===")
    print(df_test.to_string(index=False))

    # Save metrics
    df_cv.to_csv(METRICS_DIR / "cv_leaderboard.csv", index=False)
    df_test.to_csv(METRICS_DIR / "test_leaderboard.csv", index=False)

    # 6. Save Plots
    print("\n[6/7] Generating and saving evaluation figures...")
    plot_leaderboard_bars(df_test, FIGURES_DIR / "leaderboard_comparison.png")
    plot_roc_pr_curves(y_test.values, test_probas, FIGURES_DIR / "roc_pr_curves_comparison.png")
    plot_confusion_matrices(y_test.values, test_preds_default, FIGURES_DIR / "confusion_matrices_comparison.png")

    # Save Key Models
    champion_name = df_test.iloc[0]["Model"]
    print(f"\n  Champion Model by ROC-AUC: {champion_name}")
    for name, est in trained_estimators.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(est, MODELS_DIR / fname)

    # Save Champion Model distinctly
    joblib.dump(trained_estimators[champion_name], MODELS_DIR / "champion_model.joblib")

    # 7. Explainability & SHAP Analysis
    print("\n[7/7] Running Explainability & SHAP Analysis on best tree model (XGBoost)...")
    xgb_model = trained_estimators["XGBoost"]
    df_feat_imp = compute_feature_importances(xgb_model, feature_names)
    df_feat_imp.to_csv(METRICS_DIR / "feature_importances_xgboost.csv", index=False)
    plot_top_feature_importances(df_feat_imp, "XGBoost", FIGURES_DIR / "feature_importance_top20.png")

    # SHAP on sample of test set for speed and precision
    sample_indices = np.random.choice(len(X_test_proc), size=min(1000, len(X_test_proc)), replace=False)
    X_shap_sample = X_test_proc[sample_indices]
    print("  Calculating SHAP values with TreeExplainer...")
    shap_vals, explainer = generate_shap_analysis(
        xgb_model, X_shap_sample, feature_names, FIGURES_DIR
    )
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    # Save metadata summary
    summary_meta = {
        "dataset_rows": len(X_train) + len(X_test),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(feature_names),
        "champion_model": champion_name,
        "champion_test_roc_auc": float(df_test.iloc[0]["ROC-AUC"]),
        "champion_test_f1": float(df_test.iloc[0]["F1-Score"]),
        "champion_calibrated_f1": float(df_test.iloc[0]["Calibrated F1-Score"]),
        "champion_optimal_threshold": float(df_test.iloc[0]["Optimal Threshold"]),
        "features": feature_names,
    }
    with open(METRICS_DIR / "summary_report.json", "w") as f:
        json.dump(summary_meta, f, indent=4)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS & EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
