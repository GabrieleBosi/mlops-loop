"""Step 9: population stability index per feature and on the prediction distribution.

PSI in plain numpy. The formula is

    PSI = sum over bins of (actual - expected) * ln(actual / expected)

where actual and expected are the proportions of rows falling in each bin, in the batch and
in the reference respectively. It is zero when the two distributions match and it grows
without bound as a bin empties on one side. The conventional reading from credit scoring,
which configs/drift.yaml uses: below 0.10 nothing meaningful, 0.10 to 0.25 moderate, above
0.25 major.

The expression is symmetric in the two proportion vectors, but this function is not symmetric
in its arguments for numeric columns, because the bin edges are quantiles of the reference.
Swapping reference and batch re-bins the data and gives a different number. That is the
intended behaviour for monitoring, where the reference is fixed and the batch is what
arrives, and it is why the argument order is reference first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config, dataset, evaluate, features, register, report, serve, tracking
from .tracking import step
from .validate import CATEGORICAL_COLUMNS, TARGET_INT

PREDICTION_KEY = "__prediction__"


class MonitorError(Exception):
    """Raised when drift cannot be measured. Never a silent "no drift"."""


@dataclass(frozen=True)
class FeaturePSI:
    feature: str
    psi: float
    threshold: float
    kind: str
    bins: int

    @property
    def breached(self) -> bool:
        return self.psi > self.threshold

    @property
    def band(self) -> str:
        """The conventional reading, independent of the configured threshold."""
        if self.psi < 0.10:
            return "none"
        return "moderate" if self.psi <= 0.25 else "major"


@dataclass(frozen=True)
class MonitorResult:
    batch: str
    description: str
    run_id: str
    reference_split: str
    reference_rows: int
    batch_rows: int
    reference_churn: float
    batch_churn: float
    features: list[FeaturePSI]
    prediction: FeaturePSI
    model_name: str
    model_version: str
    champion_run_id: str
    batch_metrics: dict[str, float]
    reference_metrics: dict[str, float]

    @property
    def breaches(self) -> list[FeaturePSI]:
        return [item for item in self.all_psi if item.breached]

    @property
    def all_psi(self) -> list[FeaturePSI]:
        return [*self.features, self.prediction]

    @property
    def drifted(self) -> bool:
        return bool(self.breaches)


def psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
    categorical: bool = False,
    epsilon: float = 1e-6,
) -> float:
    """PSI of actual against expected.

    Numeric columns are binned on quantiles of the expected sample, so each reference bin
    holds roughly the same number of rows and the index does not depend on the units. Ties
    collapse duplicate edges, which is why the bin count is reported alongside the number.
    Categorical columns use one bin per value seen in either sample, so a category that is
    new in the batch is a bin the reference never filled.

    A bin empty on one side would send the logarithm to infinity, so empty proportions become
    epsilon. That makes a brand new category a large finite number rather than an inf, which
    is a number that can be logged, compared to a threshold and put in a table.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.size == 0 or actual.size == 0:
        raise MonitorError("PSI needs a non-empty reference and a non-empty batch")

    if categorical:
        levels = pd.unique(np.concatenate([expected, actual]))
        expected_counts = np.array([(expected == level).sum() for level in levels], dtype=float)
        actual_counts = np.array([(actual == level).sum() for level in levels], dtype=float)
    else:
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(expected.astype(float), quantiles))
        if edges.size < 2:
            # A constant reference column: one bin, and the only question is whether the
            # batch is also constant at that value.
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
        edges[0], edges[-1] = -np.inf, np.inf
        expected_counts = np.histogram(expected.astype(float), bins=edges)[0].astype(float)
        actual_counts = np.histogram(actual.astype(float), bins=edges)[0].astype(float)

    expected_share = expected_counts / expected_counts.sum()
    actual_share = actual_counts / actual_counts.sum()
    expected_share = np.where(expected_share == 0.0, epsilon, expected_share)
    actual_share = np.where(actual_share == 0.0, epsilon, actual_share)

    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def bin_count(expected: np.ndarray | pd.Series, bins: int, categorical: bool) -> int:
    """How many bins the PSI above actually used, after ties collapsed duplicate edges."""
    expected = np.asarray(expected)
    if categorical:
        return int(pd.unique(expected).size)
    edges = np.unique(np.percentile(expected.astype(float), np.linspace(0, 100, bins + 1)))
    return max(int(edges.size) - 1, 1)


def feature_psi(
    reference: pd.DataFrame, batch: pd.DataFrame, psi_cfg: dict[str, Any]
) -> list[FeaturePSI]:
    """PSI for every model input, against the thresholds named in configs/drift.yaml."""
    thresholds = psi_cfg["thresholds"]
    configured, expected = set(thresholds), set(features.FEATURE_COLUMNS)
    if configured != expected:
        raise MonitorError(
            "configs/drift.yaml psi.thresholds does not match the model's features. "
            f"missing {sorted(expected - configured)}, unexpected {sorted(configured - expected)}"
        )

    bins = int(psi_cfg.get("bins", 10))
    epsilon = float(psi_cfg.get("epsilon", 1e-6))
    results: list[FeaturePSI] = []
    for column in features.FEATURE_COLUMNS:
        categorical = column in CATEGORICAL_COLUMNS
        results.append(
            FeaturePSI(
                feature=column,
                psi=psi(reference[column], batch[column], bins, categorical, epsilon),
                threshold=float(thresholds[column]),
                kind="categorical" if categorical else "numeric",
                bins=bin_count(reference[column], bins, categorical),
            )
        )
    return sorted(results, key=lambda item: -item.psi)


def prediction_psi(
    model: Pipeline, reference: pd.DataFrame, batch: pd.DataFrame, psi_cfg: dict[str, Any]
) -> FeaturePSI:
    """PSI of the champion's predicted probabilities: the aggregate the model actually emits."""
    bins = int(psi_cfg.get("bins", 10))
    epsilon = float(psi_cfg.get("epsilon", 1e-6))
    reference_scores = evaluate.predict_proba(model, reference)
    batch_scores = evaluate.predict_proba(model, batch)
    return FeaturePSI(
        feature=PREDICTION_KEY,
        psi=psi(reference_scores, batch_scores, bins, False, epsilon),
        threshold=float(psi_cfg["prediction_threshold"]),
        kind="prediction",
        bins=bin_count(reference_scores, bins, False),
    )


def run(
    batch_name: str,
    cfg: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    prepared: dataset.Prepared | None = None,
    champion: serve.LoadedChampion | None = None,
    data_dir: str | None = None,
    source: str | None = None,
    report_dir: str | None = None,
) -> MonitorResult:
    """Score one batch with the champion and log every PSI in a monitor run."""
    cfg = cfg if cfg is not None else config.skeleton_config()
    drift = drift if drift is not None else config.drift_config()
    psi_cfg = drift["psi"]
    reference_split = drift.get("reference_split", "train")

    registry = cfg["registry"]
    if champion is None:
        try:
            champion = serve.load_champion(name=registry["name"], alias=registry["alias"])
        except register.RegistryError as exc:
            raise MonitorError(
                f"no champion to monitor with: {exc}. Run `python -m mlops_loop train` first."
            ) from exc

    git = tracking.git_info()
    with mlflow.start_run(run_name=f"monitor-{batch_name}") as active:
        mlflow.set_tags(
            {
                **git,
                "phase": "session-3",
                "pipeline": "monitor",
                "batch": batch_name,
                "model_name": champion.info.name,
                "model_version": champion.info.version,
                "champion_run_id": champion.info.run_id,
            }
        )
        if prepared is None:
            prepared = dataset.prepare(cfg=cfg, drift=drift, data_dir=data_dir, source=source)
        reference = getattr(prepared.splits, reference_split)
        batch = prepared.splits.batch(batch_name)

        with step("psi"):
            feature_results = feature_psi(reference, batch, psi_cfg)
            prediction_result = prediction_psi(champion.model, reference, batch, psi_cfg)

        mlflow.log_params(
            {
                "batch": batch_name,
                "reference_split": reference_split,
                "psi_bins": psi_cfg.get("bins", 10),
                "psi_epsilon": psi_cfg.get("epsilon", 1e-6),
            }
        )
        for item in feature_results:
            mlflow.log_metric(f"psi_{item.feature}", item.psi)
            mlflow.log_metric(f"psi_threshold_{item.feature}", item.threshold)
        mlflow.log_metric("psi_prediction", prediction_result.psi)
        mlflow.log_metric("psi_threshold_prediction", prediction_result.threshold)
        mlflow.log_metric("psi_max", max(item.psi for item in feature_results))
        mlflow.log_metric(
            "psi_breaches",
            float(sum(1 for item in [*feature_results, prediction_result] if item.breached)),
        )
        mlflow.log_metric("rows_reference", float(len(reference)))
        mlflow.log_metric("rows_batch", float(len(batch)))

        # The batch carries labels here because it is a held-out slice of a historical file.
        # Real monitoring would not have them at scoring time, which is exactly why PSI is
        # computed on inputs and predictions and the trigger never reads the batch's labels.
        with step("score_batch"):
            batch_metrics = evaluate.evaluate(champion.model, batch, prefix="batch", log=True)
            reference_metrics = evaluate.evaluate(
                champion.model, reference, prefix="reference", log=True
            )

        description = next(
            (
                entry.get("description", "")
                for entry in drift["future_batches"]
                if entry["name"] == batch_name
            ),
            "",
        )
        result = MonitorResult(
            batch=batch_name,
            description=description,
            run_id=active.info.run_id,
            reference_split=reference_split,
            reference_rows=len(reference),
            batch_rows=len(batch),
            reference_churn=float(reference[TARGET_INT].mean()),
            batch_churn=float(batch[TARGET_INT].mean()),
            features=feature_results,
            prediction=prediction_result,
            model_name=champion.info.name,
            model_version=champion.info.version,
            champion_run_id=champion.info.run_id,
            batch_metrics=batch_metrics,
            reference_metrics=reference_metrics,
        )
        mlflow.set_tag("drift", "yes" if result.drifted else "no")
        mlflow.log_dict(
            {
                "batch": batch_name,
                "drifted": result.drifted,
                "breaches": [item.feature for item in result.breaches],
                "psi": {item.feature: item.psi for item in result.all_psi},
                "thresholds": {item.feature: item.threshold for item in result.all_psi},
            },
            "psi.json",
        )
        written = report.write_drift_report(result, directory=report_dir)
        mlflow.log_artifact(str(written), artifact_path="reports")

    return result


def format_result(result: MonitorResult) -> str:
    """The table the CLI prints for one batch."""
    lines = [
        "",
        f"monitor: batch {result.batch} ({result.batch_rows:,} rows) against "
        f"{result.reference_split} ({result.reference_rows:,} rows)",
        f"  monitor run   {result.run_id}",
        f"  champion      {result.model_name} version {result.model_version}",
        "",
        f"  {'feature':<26}{'PSI':>10}{'threshold':>11}{'bins':>6}   band     result",
    ]
    for item in result.all_psi:
        name = "prediction distribution" if item.feature == PREDICTION_KEY else item.feature
        lines.append(
            f"  {name:<26}{item.psi:>10.4f}{item.threshold:>11.2f}{item.bins:>6}"
            f"   {item.band:<8} {'BREACH' if item.breached else 'ok'}"
        )
    lines += [
        "",
        f"  churn rate    reference {result.reference_churn:.4f}  batch {result.batch_churn:.4f}",
        f"  champion PR-AUC  reference {result.reference_metrics['pr_auc']:.4f}  "
        f"batch {result.batch_metrics['pr_auc']:.4f}",
        f"  verdict       {'DRIFTED' if result.drifted else 'no drift'}"
        f" ({len(result.breaches)} breach(es))",
        "",
    ]
    return "\n".join(lines)
