"""
Feature Engineering module for Bank Marketing Prediction.
Implements a Scikit-Learn compatible Transformer that engineers domain-specific
interaction features, ratios, and transformations with zero data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import MONTH_MAPPING


class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that generates interaction features
    and domain-informed variables for the bank marketing dataset.
    """

    def __init__(self, include_cyclical: bool = True):
        self.include_cyclical = include_cyclical
        self.education_map = {
            "unknown": 0,
            "primary": 1,
            "secondary": 2,
            "tertiary": 3,
        }

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer - no statistics to leak across train/test
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Ordinal Education conversion
        edu_num = X["education"].map(self.education_map).fillna(0).astype(float)
        X["education_ordinal"] = edu_num

        # 2. Key Interactions: Duration, Balance, Age, Education
        # Duration per contact attempt in this campaign
        X["duration_per_campaign"] = X["duration"] / (X["campaign"] + 1e-5)

        # Wealth / Age ratio (financial maturity indicator)
        X["balance_to_age"] = X["balance"] / (X["age"] + 1e-5)

        # Interaction: Age & Education (life stage vs credential level)
        X["age_education_interact"] = X["age"] * edu_num

        # Interaction: Duration & Balance (high engagement from affluent client)
        balance_sign_log = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
        X["duration_balance_interact"] = balance_sign_log * np.log1p(X["duration"])

        # 3. Financial Burden / Debt Profile
        has_housing = (X["housing"].str.lower() == "yes").astype(int)
        has_loan = (X["loan"].str.lower() == "yes").astype(int)
        X["total_loans"] = has_housing + has_loan
        X["is_debt_free"] = ((has_housing == 0) & (has_loan == 0)).astype(int)
        X["is_negative_balance"] = (X["balance"] < 0).astype(int)

        # 4. Previous Campaign Engagement (pdays = -1 means never contacted before)
        X["pdays_contacted"] = (X["pdays"] != -1).astype(int)
        X["pdays_cleaned"] = np.where(X["pdays"] == -1, 0, X["pdays"])
        X["campaign_to_previous_ratio"] = X["campaign"] / (X["previous"] + 1.0)

        # 5. Cyclical Date Encodings
        if self.include_cyclical:
            month_num = X["month"].str.lower().map(MONTH_MAPPING).fillna(6).astype(float)
            X["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
            X["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)

            day_num = X["day"].astype(float)
            X["day_sin"] = np.sin(2 * np.pi * day_num / 31.0)
            X["day_cos"] = np.cos(2 * np.pi * day_num / 31.0)

        return X
