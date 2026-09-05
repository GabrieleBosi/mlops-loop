"""Step 4: the feature matrix. Fixed categories, so the column set never moves."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .tracking import code_hash
from .validate import CATEGORICAL_COLUMNS, CATEGORIES, NUMERIC_COLUMNS

# customerID identifies a row, it does not describe one. Churn and churn are the target.
# Neither is ever a feature.
FEATURE_COLUMNS: list[str] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Scale the four numerics, one-hot the fifteen categoricals against fixed categories.

    categories= is pinned to the validated sets rather than learned from the fit data. A
    model trained on the reference split, which has no fibre-optic customer, still gets an
    InternetService_Fiber optic column of zeros, so the champion and a challenger trained on
    the future batch produce the same matrix shape and can be compared on the same holdout.
    handle_unknown="error" means a category the schema never saw stops the prediction.
    """
    encoder = OneHotEncoder(
        categories=[CATEGORIES[column] for column in CATEGORICAL_COLUMNS],
        handle_unknown="error",
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", encoder, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The model input: the feature columns only, in a fixed order."""
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"feature columns missing from frame: {missing}")
    return frame[FEATURE_COLUMNS].copy()


def encoded_columns(fitted: ColumnTransformer | Pipeline) -> list[str]:
    """The column names the fitted preprocessor produces."""
    transformer = fitted.named_steps["preprocessor"] if isinstance(fitted, Pipeline) else fitted
    return [str(name) for name in transformer.get_feature_names_out()]


def feature_code_hash() -> str:
    """sha256 of this file. Any change to the encoding changes the hash on the run."""
    return code_hash(Path(__file__))


def log_features(fitted: ColumnTransformer | Pipeline) -> dict[str, list[str]]:
    """Log the feature-code hash and both column lists as a run artifact."""
    columns = {
        "input_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoded_columns": encoded_columns(fitted),
    }
    mlflow.log_param("feature_code_hash", feature_code_hash())
    mlflow.log_param("n_encoded_features", len(columns["encoded_columns"]))
    mlflow.log_dict(columns, "feature_columns.json")
    return columns
