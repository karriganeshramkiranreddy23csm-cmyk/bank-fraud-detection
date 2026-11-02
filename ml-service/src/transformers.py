import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible outlier capper / winsorizer.
    Bounds are learned strictly on the training set during .fit() to prevent data leakage.
    During .transform(), values outside the learned bounds are capped/clipped.
    
    Supports:
      - 'percentile': Winsorization using lower_percentile and upper_percentile.
      - 'iqr': Boxplot rule using Q1 - k*IQR and Q3 + k*IQR (e.g., k=1.5 or k=3.0).
      - 'none': Passthrough with no bounds.
    """
    def __init__(
        self,
        strategy="percentile",
        lower_percentile=1.0,
        upper_percentile=99.0,
        iqr_multiplier=1.5,
        columns=None
    ):
        self.strategy = strategy
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.iqr_multiplier = iqr_multiplier
        self.columns = columns
        self.bounds_ = {}
        self.last_clipped_stats_ = {}

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        cols_to_fit = self.columns if self.columns is not None else list(X_df.columns)

        self.bounds_ = {}
        for col in cols_to_fit:
            if col not in X_df.columns:
                continue
            series = X_df[col].dropna().values
            if len(series) == 0:
                self.bounds_[col] = {"lower": -np.inf, "upper": np.inf}
                continue

            if self.strategy == "percentile":
                lower = float(np.percentile(series, self.lower_percentile))
                upper = float(np.percentile(series, self.upper_percentile))
            elif self.strategy == "iqr":
                q25 = float(np.percentile(series, 25))
                q75 = float(np.percentile(series, 75))
                iqr = q75 - q25
                lower = q25 - (self.iqr_multiplier * iqr)
                upper = q75 + (self.iqr_multiplier * iqr)
            elif self.strategy == "none":
                lower = -np.inf
                upper = np.inf
            else:
                raise ValueError(f"Unknown strategy: '{self.strategy}'. Choose 'percentile', 'iqr', or 'none'.")

            self.bounds_[col] = {"lower": lower, "upper": upper}

        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        X_df = self._to_dataframe(X).copy()
        n_rows = len(X_df)
        stats = {}

        for col, bound in self.bounds_.items():
            if col not in X_df.columns:
                continue
            lower, upper = bound["lower"], bound["upper"]
            vals = X_df[col].values

            lower_clipped_mask = vals < lower
            upper_clipped_mask = vals > upper

            lower_count = int(np.sum(lower_clipped_mask))
            upper_count = int(np.sum(upper_clipped_mask))
            total_clipped = lower_count + upper_count
            pct_clipped = round((total_clipped / n_rows) * 100, 4) if n_rows > 0 else 0.0

            stats[col] = {
                "lower_bound": round(lower, 4) if not np.isneginf(lower) else None,
                "upper_bound": round(upper, 4) if not np.isposinf(upper) else None,
                "lower_clipped_count": lower_count,
                "upper_clipped_count": upper_count,
                "total_clipped_count": total_clipped,
                "clipped_percentage": pct_clipped,
            }

            if self.strategy != "none":
                X_df[col] = np.clip(vals, lower, upper)

        self.last_clipped_stats_ = stats
        return X_df if is_df else X_df.values

    def get_report(self, X=None):
        """
        Returns a detailed clipping audit report, optionally evaluated against a specific dataset X.
        """
        if X is not None:
            # Run transform on a copy to populate stats for X without altering state
            _ = self.transform(X)

        return {
            "strategy": self.strategy,
            "params": {
                "lower_percentile": self.lower_percentile,
                "upper_percentile": self.upper_percentile,
                "iqr_multiplier": self.iqr_multiplier,
            },
            "columns": self.last_clipped_stats_
        }

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            col_names = self.columns if self.columns and len(self.columns) == X.shape[1] else [f"col_{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=col_names)
        return pd.DataFrame(X)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p (natural log(1 + x)) transformation to non-negative financial values (e.g., Amount).
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        X_df = (X if is_df else pd.DataFrame(X)).copy()
        cols = self.columns if self.columns is not None else list(X_df.columns)

        for col in cols:
            if col in X_df.columns:
                # Ensure values >= 0 before log1p
                vals = np.maximum(0, X_df[col].values.astype(float))
                X_df[col] = np.log1p(vals)

        return X_df if is_df else X_df.values
