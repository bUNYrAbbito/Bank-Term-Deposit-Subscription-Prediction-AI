"""
Configuration module for Bank-Term Deposit Subscription Prediction.
Defines paths, feature column mappings, hyperparameters, and project constants.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "bank-full.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Seed & Cross-Validation
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20

# Target
TARGET_COL = "y"
TARGET_MAPPING = {"no": 0, "yes": 1}

# Column categories
NUMERICAL_COLS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

# Ordered categorical columns (ordinal)
ORDINAL_COLS = ["education"]
ORDINAL_MAPPING = {
    "education": ["unknown", "primary", "secondary", "tertiary"]
}

# Nominal categorical columns (one-hot)
NOMINAL_COLS = [
    "job",
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Month mapping to numeric for cyclical encoding
MONTH_MAPPING = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}
