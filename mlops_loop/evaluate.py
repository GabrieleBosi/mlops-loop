"""Step 6: the four metrics and the calibration plot."""

from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from .features import feature_frame  # noqa: E402
from .validate import TARGET_INT  # noqa: E402

METRIC_NAMES = ("roc_auc", "pr_auc", "recall_at_precision_50", "brier")


class EvaluationError(Exception):
    """Raised when a metric cannot be computed. Partial results are never returned."""


def predict_proba(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Probability of churn for each row."""
    return model.predict_proba(feature_frame(frame))[:, 1]


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray, floor: float = 0.5) -> float:
    """Highest recall reachable at a threshold whose precision is at least the floor.

    This is the number the business cares about: how many churners can be caught if at
    least half of the flagged customers must really be churning. Returns 0.0 when no
    threshold reaches the precision floor, which is a real answer, not a missing one.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    reachable = recall[precision >= floor]
    return float(reachable.max()) if reachable.size else 0.0


def evaluate(model: Pipeline, frame: pd.DataFrame, prefix: str, log: bool = True) -> dict[str, float]:
    """Score a split. Stops if the split has one class or any metric comes back NaN."""
    if frame.empty:
        raise EvaluationError(f"cannot evaluate '{prefix}': the frame is empty")
    y_true = frame[TARGET_INT].to_numpy()
    if len(np.unique(y_true)) < 2:
        raise EvaluationError(
            f"cannot evaluate '{prefix}': the target has one class only, {np.unique(y_true)}"
        )

    scores = predict_proba(model, frame)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_precision_50": recall_at_precision(y_true, scores, floor=0.5),
        "brier": float(brier_score_loss(y_true, scores)),
    }
    bad = [name for name, value in metrics.items() if not math.isfinite(value)]
    if bad:
        raise EvaluationError(f"'{prefix}' produced non-finite metrics: {bad}")

    if log:
        mlflow.log_metrics({f"{prefix}_{name}": value for name, value in metrics.items()})
    return metrics


def calibration_plot(
    model: Pipeline, frame: pd.DataFrame, prefix: str, n_bins: int = 10, log: bool = True
) -> Any:
    """Reliability curve on a split, logged as calibration_<prefix>.png.

    Quantile bins, so every point rests on the same number of customers.
    """
    y_true = frame[TARGET_INT].to_numpy()
    scores = predict_proba(model, frame)
    observed, predicted = calibration_curve(y_true, scores, n_bins=n_bins, strategy="quantile")

    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    axes.plot(predicted, observed, marker="o", label=prefix)
    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed churn rate")
    axes.set_title(f"calibration on {prefix} ({len(frame)} rows, {n_bins} quantile bins)")
    axes.legend(loc="upper left")
    figure.tight_layout()

    if log:
        mlflow.log_figure(figure, f"calibration_{prefix}.png")
    plt.close(figure)
    return figure
