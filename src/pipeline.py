"""
Pipeline module for Bank Term Deposit Prediction.
Assembles the complete scikit-learn preprocessing pipeline including feature engineering,
robust scaling of skewed numerical features, and one-hot encoding of nominal categories.
"""

from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from src.feature_engineering import BankFeatureEngineer


def get_column_definitions() -> Tuple[List[str], List[str]]:
    """
    Returns the nominal categorical columns and the post-feature-engineering numeric columns.
    """
    nominal_cols = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    numeric_cols = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "education_ordinal",
        "duration_per_campaign",
        "balance_to_age",
        "age_education_interact",
        "duration_balance_interact",
        "total_loans",
        "is_debt_free",
        "is_negative_balance",
        "pdays_contacted",
        "pdays_cleaned",
        "campaign_to_previous_ratio",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
    ]
    return nominal_cols, numeric_cols


def build_preprocessor() -> Pipeline:
    """
    Builds and returns the full leak-free preprocessing pipeline.
    """
    nominal_cols, numeric_cols = get_column_definitions()

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", BankFeatureEngineer()),
            ("col_transformer", col_transformer),
        ]
    )
    return full_pipeline


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """
    Extract readable feature names from the fitted column transformer.
    """
    col_transformer = fitted_pipeline.named_steps["col_transformer"]
    nominal_cols, numeric_cols = get_column_definitions()

    feature_names = list(numeric_cols)
    cat_encoder = col_transformer.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(nominal_cols)
    feature_names.extend(cat_feature_names)
    return feature_names
