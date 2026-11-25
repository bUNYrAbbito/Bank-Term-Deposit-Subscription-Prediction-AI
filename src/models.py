"""
Models module for Bank Term Deposit Prediction.
Instantiates baselines, tree-based models, gradient boosting architectures,
deep learning MLP, and ensemble classifiers with robust class imbalance handling.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.config import RANDOM_STATE


def get_candidate_models(scale_pos_weight: float = 7.55) -> Dict[str, Any]:
    """
    Returns a dictionary of all initialized candidate models.
    """
    models = {
        # 1. Baseline: Logistic Regression
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        # 2. Baseline: Gaussian Naive Bayes
        "Naive Bayes": GaussianNB(),
        # 3. Support Vector Machine (Probability-calibrated SGD with log-loss)
        "Support Vector Machine": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=1500,
            random_state=RANDOM_STATE,
        ),
        # 4. Tree-based: Random Forest
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # 5. Sklearn Gradient Boosting
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # 6. Advanced Boosting: XGBoost
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # 7. Advanced Boosting: CatBoost
        "CatBoost": CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.06,
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=RANDOM_STATE,
        ),
        # 8. Deep Learning: Multi-Layer Perceptron (MLP)
        "Deep Learning (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def build_stacking_classifier(
    base_models: Dict[str, Any], final_estimator=None
) -> StackingClassifier:
    """
    Builds a Stacking Classifier using top diverse base estimators and
    a regularized Logistic Regression meta-classifier.
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=1000, random_state=RANDOM_STATE
        )

    estimators = [
        ("xgb", base_models["XGBoost"]),
        ("catboost", base_models["CatBoost"]),
        ("rf", base_models["Random Forest"]),
        ("hgb", base_models["Gradient Boosting"]),
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stacking_clf
