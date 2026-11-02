import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_STATE = 42

# Column Definitions
RAW_NUMERICAL_COLS = ["Time", "Amount"]
PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = RAW_NUMERICAL_COLS + PCA_COLS
TARGET_COL = "Class"

# Split Ratios (Stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Default Costs for Cost-Sensitive Threshold Selection
DEFAULT_FN_COST = 100.0  # Cost of missing a fraudulent transaction
DEFAULT_FP_COST = 5.0    # Cost of false alarm (investigation / friction)
