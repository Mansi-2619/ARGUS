"""
preprocess.py  -  split the data and build the Preprocessor used for BOTH
training and inference.

Pipeline inside Preprocessor.transform():
    check columns -> reorder -> numeric -> reject NaN/inf -> log1p on saved
    skewed columns -> RobustScaler (fitted on training data only)
"""
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from config import ARTIFACT_DIR, LABEL_COL, RANDOM_SEED, SKEW_THRESHOLD, TEST_SIZE

BENIGN_LABEL = "BENIGN"


# ----------------------------------------------------------------------------
# Splitting  (one split, shared by the binary and multiclass tasks)
# ----------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """Stratified split on the multiclass label. Returns string labels."""
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )


def binary_target(y: pd.Series) -> pd.Series:
    """1 = attack, 0 = benign. Defined explicitly so 'positive' always means attack."""
    return (y != BENIGN_LABEL).astype("int8")


# ----------------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------------
class Preprocessor:
    def __init__(self):
        self.feature_cols = None       # ordered column list seen in training
        self.log_cols = None           # columns that get log1p
        self.skipped_negative = None   # skewed but contain negatives -> not logged
        self.zero_iqr_cols = None      # columns RobustScaler cannot scale
        self.scaler = None

    # -- helpers -------------------------------------------------------------
    def _check_fitted(self):
        if self.scaler is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit() or load().")

    @staticmethod
    def _check_finite(X: pd.DataFrame):
        bad = ~np.isfinite(X.to_numpy())
        if bad.any():
            per_col = pd.Series(bad.sum(axis=0), index=X.columns)
            per_col = per_col[per_col > 0].sort_values(ascending=False)
            raise ValueError(
                f"{int(bad.any(axis=1).sum()):,} row(s) contain NaN/inf. "
                f"Worst columns:\n{per_col.head(5).to_string()}"
            )

    def _apply_log(self, X: pd.DataFrame) -> pd.DataFrame:
        """Used by BOTH fit and transform, so they can never differ."""
        X = X.copy()
        X[self.log_cols] = np.log1p(X[self.log_cols].clip(lower=0))
        return X

    # -- fit -----------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame):
        self.feature_cols = list(X_train.columns)
        X = X_train[self.feature_cols].astype("float32")
        self._check_finite(X)

        skew = X.skew()
        has_negative = X.min() < 0
        skewed = skew > SKEW_THRESHOLD            # NaN skew (constant column) -> False
        self.log_cols = [c for c in self.feature_cols if skewed[c] and not has_negative[c]]
        self.skipped_negative = [c for c in self.feature_cols if skewed[c] and has_negative[c]]

        X_logged = self._apply_log(X)
        iqr = X_logged.quantile(0.75) - X_logged.quantile(0.25)
        self.zero_iqr_cols = [c for c in self.feature_cols if iqr[c] == 0]

        self.scaler = RobustScaler().fit(X_logged)
        return self

    # -- transform -----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()

        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} required column(s): {missing[:10]}")

        # Reorder to the saved order; extra columns are ignored.
        X = X[self.feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        self._check_finite(X)

        n_neg = int((X[self.log_cols] < 0).to_numpy().sum())
        if n_neg:
            warnings.warn(
                f"{n_neg:,} negative value(s) in log-transformed columns were clipped to 0."
            )

        X_scaled = self.scaler.transform(self._apply_log(X))
        return pd.DataFrame(X_scaled, columns=self.feature_cols, index=X.index)

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        self._check_fitted()
        joblib.dump(
            {
                "feature_cols": self.feature_cols,
                "log_cols": self.log_cols,
                "skipped_negative": self.skipped_negative,
                "zero_iqr_cols": self.zero_iqr_cols,
                "scaler": self.scaler,
                "skew_threshold": SKEW_THRESHOLD,
                "sklearn_version": sklearn.__version__,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        """Only load files you created yourself (pickle can run code)."""
        d = joblib.load(path)
        if d["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"Saved with scikit-learn {d['sklearn_version']}, "
                f"running {sklearn.__version__}. Results may differ."
            )
        obj = cls()
        obj.feature_cols = d["feature_cols"]
        obj.log_cols = d["log_cols"]
        obj.skipped_negative = d["skipped_negative"]
        obj.zero_iqr_cols = d["zero_iqr_cols"]
        obj.scaler = d["scaler"]
        return obj

    def summary(self):
        print(f"Features:                     {len(self.feature_cols)}")
        print(f"Log-transformed (skew>{SKEW_THRESHOLD}):  {len(self.log_cols)}")
        print(f"Skewed but have negatives:    {self.skipped_negative}")
        print(f"Zero-IQR columns (unscaled):  {len(self.zero_iqr_cols)} -> {self.zero_iqr_cols}")


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def run_checks(df: pd.DataFrame, save_path=None):
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")
    print("Binary positive rate (train / test):",
          round(binary_target(y_train).mean(), 4), round(binary_target(y_test).mean(), 4))

    pre = Preprocessor().fit(X_train)
    pre.summary()
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)

    # Check 1: finite output, correct columns
    assert np.isfinite(Xt_train.to_numpy()).all(), "non-finite values after transform"
    assert list(Xt_train.columns) == pre.feature_cols, "column order wrong"
    print("PASS 1: output is finite and columns match feature_cols")

    # Check 2: save -> load -> same output
    path = save_path or (ARTIFACT_DIR / "preprocessor.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    pre.save(path)
    reloaded = Preprocessor.load(path)
    assert np.allclose(reloaded.transform(X_test).to_numpy(), Xt_test.to_numpy()), \
        "reloaded preprocessor gives different output"
    print("PASS 2: saved and reloaded preprocessor reproduces the same output")

    # Check 3: column order of the input must not matter
    cols = list(X_test.columns)
    np.random.default_rng(0).shuffle(cols)
    assert np.allclose(pre.transform(X_test[cols]).to_numpy(), Xt_test.to_numpy()), \
        "output changed when input columns were shuffled"
    print("PASS 3: shuffled input columns give identical output")

    # Check 4: a missing column is rejected with a clear error
    try:
        pre.transform(X_test.drop(columns=[pre.feature_cols[0]]))
        raise AssertionError("missing column was not rejected")
    except ValueError:
        print("PASS 4: missing column raises ValueError")


if __name__ == "__main__":
    from load_data import load_raw
    run_checks(load_raw())