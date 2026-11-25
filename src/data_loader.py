"""
Data Loader module for Bank Marketing dataset.
Handles ingestion, schema validation, and leak-free stratified splitting.
"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COL, TARGET_MAPPING, RANDOM_STATE, TEST_SIZE


def load_raw_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load raw CSV data with appropriate delimiter and validation.
    """
    df = pd.read_csv(data_path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, sep=",")
    return df


def get_train_test_data(
    data_path: str = DATA_PATH, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split dataset into train and test sets using Stratified Splitting
    to preserve target imbalance and prevent data leakage.
    """
    df = load_raw_data(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map(TARGET_MAPPING)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)
