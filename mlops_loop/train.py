"""Step 5: fit one pipeline. Session 2 wraps this in the sweep; the signature stays."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, feature_frame
from .split import assert_holdout_unseen
from .validate import TARGET_INT


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    """Preprocessor plus logistic regression, as one estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(**params)),
        ]
    )


def fit_model(
    train_frame: pd.DataFrame,
    params: dict[str, Any],
    holdout: pd.DataFrame | None = None,
    log: bool = True,
) -> Pipeline:
    """Fit on train_frame after checking the holdout is not in it.

    holdout is passed in so the leak check runs against real ids rather than a promise.
    """
    if holdout is not None:
        assert_holdout_unseen(train_frame, holdout)

    pipeline = build_pipeline(params)
    pipeline.fit(feature_frame(train_frame), train_frame[TARGET_INT])

    if log:
        mlflow.log_params({f"model_{key}": value for key, value in params.items()})
        mlflow.log_param("model_class", "LogisticRegression")
        mlflow.log_metric("rows_fit", float(len(train_frame)))
    return pipeline
