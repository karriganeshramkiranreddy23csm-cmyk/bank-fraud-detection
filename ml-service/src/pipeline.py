import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.transformers import OutlierCapper, Log1pTransformer
from src.config import RANDOM_STATE, RAW_NUMERICAL_COLS, PCA_COLS, FEATURE_COLS


def create_preprocessor(capper_strategy="percentile", scaler_type="robust", capper_cols=None):
    """
    Constructs the preprocessor pipeline:
    1. OutlierCapper on configured columns (defaults to ['Time', 'Amount']).
    2. Optional Log1p on Amount (if scaler_type is log1p_robust).
    3. Scaler on ['Time', 'Amount'] (RobustScaler or StandardScaler).
    4. Passthrough on V1..V28.
    """
    if capper_cols is None:
        capper_cols = ["Time", "Amount"]

    capper = OutlierCapper(
        strategy=capper_strategy,
        columns=capper_cols,
        lower_percentile=1.0,
        upper_percentile=99.0,
        iqr_multiplier=1.5
    )

    # Scaler definition
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "log1p_robust":
        scaler = ImbPipeline([
            ("log1p", Log1pTransformer(columns=["Amount"])),
            ("robust", RobustScaler())
        ])
    elif scaler_type == "none":
        scaler = "passthrough"
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    # ColumnTransformer scales RAW_NUMERICAL_COLS and passes remainder (PCA_COLS) through
    col_transformer = ColumnTransformer(
        transformers=[
            ("num_scaler", scaler, RAW_NUMERICAL_COLS)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    return capper, col_transformer


def create_classifier(model_type="xgb", imbalance_strategy="balanced", pos_weight=1.0, random_state=RANDOM_STATE):
    """
    Instantiates classifier configured with imbalance parameter (class_weight / scale_pos_weight).
    """
    is_balanced = (imbalance_strategy == "balanced")

    if model_type == "dummy":
        return DummyClassifier(strategy="most_frequent")

    elif model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if is_balanced else None,
            random_state=random_state
        )

    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced" if is_balanced else None,
            random_state=random_state,
            n_jobs=-1
        )

    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=pos_weight if is_balanced else 1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

    elif model_type == "lgb":
        return LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            class_weight="balanced" if is_balanced else None,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose from 'dummy', 'lr', 'rf', 'xgb', 'lgb'.")


def build_fraud_pipeline(
    model_type="xgb",
    capper_strategy="percentile",
    scaler_type="robust",
    imbalance_strategy="balanced",
    capper_cols=None,
    pos_weight=1.0,
    random_state=RANDOM_STATE
):
    """
    Builds an end-to-end imblearn.pipeline.Pipeline:
    [capper] -> [scaler_col_transformer] -> [resampler (if SMOTE or RUS)] -> [classifier]
    Resampling is strictly applied during fit() on train data only; never on validation or test.
    """
    capper, col_transformer = create_preprocessor(
        capper_strategy=capper_strategy,
        scaler_type=scaler_type,
        capper_cols=capper_cols
    )

    steps = [
        ("capper", capper),
        ("preprocessor", col_transformer),
    ]

    # Imbalance resampler inside pipeline
    if imbalance_strategy == "smote":
        # Over-sample minority class to 20% of majority in train fold
        steps.append(("resampler", SMOTE(sampling_strategy=0.2, random_state=random_state)))
    elif imbalance_strategy == "rus":
        # Under-sample majority class to 10% ratio in train fold
        steps.append(("resampler", RandomUnderSampler(sampling_strategy=0.1, random_state=random_state)))
    elif imbalance_strategy in ("balanced", "none"):
        # No resampling step needed in pipeline
        pass
    else:
        raise ValueError(f"Unknown imbalance_strategy: {imbalance_strategy}")

    clf = create_classifier(
        model_type=model_type,
        imbalance_strategy=imbalance_strategy,
        pos_weight=pos_weight,
        random_state=random_state
    )
    steps.append(("classifier", clf))

    return ImbPipeline(steps)
