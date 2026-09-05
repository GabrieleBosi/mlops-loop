"""Step 10: the retraining trigger and the champion versus challenger decision.

Drift alone never promotes anything. A breached PSI only earns the right to train a
challenger; the challenger then has to beat the champion by more than a margin, on the fixed
holdout, with both models scored in the same run against the same rows. Both outcomes are
written down, because a rejection is as much a result as a promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from . import config, evaluate, features, monitor, register, report, tracking, train
from .split import HoldoutBudget, Splits, assert_holdout_unseen
from .tracking import step

PROMOTED = "promoted"
REJECTED = "rejected"


class RetrainError(Exception):
    """Raised when a challenger cannot be trained or compared. Never a silent promotion."""


@dataclass
class Decision:
    batch: str
    run_id: str
    monitor_run_id: str
    verdict: str
    metric: str
    margin: float
    improvement: float
    champion_version: str
    champion_run_id: str
    champion_metrics: dict[str, float]
    challenger_run_id: str
    challenger_metrics: dict[str, float]
    challenger_version: str | None
    challenger_rows: int
    params: dict[str, Any]
    family: str
    holdout_spends: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.verdict == PROMOTED


def champion_config(run_id: str) -> tuple[str, dict[str, Any]]:
    """Recover the family and hyperparameters the champion was fitted with.

    They are read back off the champion's own run rather than out of a config file, so the
    challenger is the same model specification trained on more data and the comparison
    isolates the data change. MLflow stores params as strings; yaml.safe_load turns them
    back into the numbers, booleans and nulls sklearn expects.
    """
    try:
        params = MlflowClient().get_run(run_id).data.params
    except Exception as exc:
        raise RetrainError(f"cannot read the champion's run {run_id}: {exc}") from exc

    family = params.get("model_family")
    if not family:
        raise RetrainError(
            f"champion run {run_id} has no model_family param, so its configuration cannot "
            "be reproduced. Retrain the champion with a current version of train.py."
        )

    recovered: dict[str, Any] = {}
    for key, value in params.items():
        if not key.startswith("model_") or key in ("model_class", "model_family"):
            continue
        parsed = yaml.safe_load(value)
        recovered[key[len("model_") :]] = None if parsed == "None" else parsed
    return family, recovered


def fit_challenger(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    holdout: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    log: bool = True,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit the champion's configuration on reference plus the drifted batch.

    The fixed holdout is never trained on, and this is where that is enforced rather than
    assumed: assert_holdout_unseen compares customerIDs and raises before anything is fitted.
    """
    fit_frame = pd.concat([reference, batch], ignore_index=True)
    assert_holdout_unseen(fit_frame, holdout)
    model = train.fit_model(fit_frame, params, holdout=holdout, family=family, log=log)
    return model, fit_frame


def run(
    result: monitor.MonitorResult,
    splits: Splits,
    champion_model: Pipeline,
    cfg: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    report_dir: str | None = None,
) -> Decision:
    """Train a challenger, compare it with the champion, promote or reject, log either way."""
    cfg = cfg if cfg is not None else config.skeleton_config()
    drift = drift if drift is not None else config.drift_config()
    promotion = drift["promotion"]
    metric = promotion.get("metric", "pr_auc")
    margin = float(promotion["margin"])
    split_name = promotion.get("split", "holdout")

    reference = getattr(splits, result.reference_split)
    batch = splits.batch(result.batch)
    holdout = getattr(splits, split_name)
    family, params = champion_config(result.champion_run_id)
    git = tracking.git_info()

    # Two looks at the holdout, one per model, both after the decision rule is fixed. The
    # budget is what stops a third from creeping in.
    budget = HoldoutBudget(holdout, budget=2)

    with mlflow.start_run(run_name=f"challenger-{result.batch}") as challenger_run:
        challenger_run_id = challenger_run.info.run_id
        mlflow.set_tags(
            {
                **git,
                "phase": "session-3",
                "pipeline": "challenger",
                "batch": result.batch,
                "monitor_run_id": result.run_id,
                "champion_run_id": result.champion_run_id,
            }
        )
        with step("fit_challenger"):
            challenger, fit_frame = fit_challenger(reference, batch, holdout, family, params)
        features.log_features(challenger)
        mlflow.log_params(
            {
                "batch": result.batch,
                "trained_on": f"{result.reference_split}+{result.batch}",
                "rows_reference": len(reference),
                "rows_batch": len(batch),
            }
        )
        with step("score_challenger"):
            challenger_metrics = evaluate.evaluate(
                challenger,
                budget.spend(f"challenger {challenger_run_id}"),
                prefix=split_name,
                log=True,
            )
            evaluate.calibration_plot(challenger, holdout, prefix=split_name)

    champion_metrics = evaluate.evaluate(
        champion_model,
        budget.spend(f"champion version {result.model_version}"),
        prefix=split_name,
        log=False,
    )
    budget.assert_fully_spent()

    improvement = challenger_metrics[metric] - champion_metrics[metric]
    verdict = PROMOTED if improvement > margin else REJECTED

    with mlflow.start_run(run_name=f"promotion_decision-{result.batch}") as decision_run:
        decision_run_id = decision_run.info.run_id
        mlflow.set_tags(
            {
                **git,
                "phase": "session-3",
                "pipeline": "promotion_decision",
                "batch": result.batch,
                "verdict": verdict,
                "monitor_run_id": result.run_id,
                "champion_run_id": result.champion_run_id,
                "challenger_run_id": challenger_run_id,
            }
        )
        mlflow.log_params(
            {
                "batch": result.batch,
                "decision_metric": metric,
                "decision_split": split_name,
                "promotion_margin": margin,
                "champion_version": result.model_version,
                "champion_run_id": result.champion_run_id,
                "challenger_run_id": challenger_run_id,
                "monitor_run_id": result.run_id,
                "challenger_family": family,
            }
        )
        mlflow.log_metrics(
            {
                **{f"champion_{split_name}_{k}": v for k, v in champion_metrics.items()},
                **{f"challenger_{split_name}_{k}": v for k, v in challenger_metrics.items()},
                f"improvement_{metric}": improvement,
                "margin": margin,
                "promoted": 1.0 if verdict == PROMOTED else 0.0,
            }
        )

        challenger_version: str | None = None
        if verdict == PROMOTED:
            with step("promote"):
                with mlflow.start_run(run_id=challenger_run_id, nested=True):
                    registered = register.register(
                        challenger,
                        fit_frame,
                        name=cfg["registry"]["name"],
                        alias=cfg["registry"]["alias"],
                        promote=True,
                    )
                challenger_version = registered.version
                mlflow.set_tags(
                    {
                        "promoted_version": challenger_version,
                        "demoted_version": registered.previous_version or "none",
                    }
                )

        decision = Decision(
            batch=result.batch,
            run_id=decision_run_id,
            monitor_run_id=result.run_id,
            verdict=verdict,
            metric=metric,
            margin=margin,
            improvement=improvement,
            champion_version=result.model_version,
            champion_run_id=result.champion_run_id,
            champion_metrics=champion_metrics,
            challenger_run_id=challenger_run_id,
            challenger_metrics=challenger_metrics,
            challenger_version=challenger_version,
            challenger_rows=len(fit_frame),
            params=params,
            family=family,
            holdout_spends=budget.reasons,
        )
        mlflow.log_dict(_as_dict(decision, split_name), "promotion_decision.json")

    return decision


def _as_dict(decision: Decision, split_name: str) -> dict[str, Any]:
    return {
        "batch": decision.batch,
        "verdict": decision.verdict,
        "rule": (
            f"promote the challenger only if {decision.metric} on the fixed {split_name} "
            f"improves by more than {decision.margin}"
        ),
        "metric": decision.metric,
        "margin": decision.margin,
        "improvement": decision.improvement,
        "champion": {
            "version": decision.champion_version,
            "run_id": decision.champion_run_id,
            split_name: decision.champion_metrics,
        },
        "challenger": {
            "run_id": decision.challenger_run_id,
            "version": decision.challenger_version,
            "family": decision.family,
            "params": decision.params,
            "trained_on_rows": decision.challenger_rows,
            split_name: decision.challenger_metrics,
        },
        "holdout_looks": decision.holdout_spends,
    }


def format_decision(decision: Decision) -> str:
    """The comparison the CLI prints. Reads the same whichever way it went."""
    lines = [
        f"  promotion decision for {decision.batch}",
        f"    decision run  {decision.run_id}",
        f"    champion      version {decision.champion_version} ({decision.champion_run_id})",
        f"    challenger    {decision.family} on {decision.challenger_rows:,} rows "
        f"({decision.challenger_run_id})",
        "",
        f"    {'metric':<26}{'champion':>10}{'challenger':>12}{'delta':>10}",
    ]
    for key in sorted(decision.champion_metrics):
        champ = decision.champion_metrics[key]
        chall = decision.challenger_metrics[key]
        lines.append(f"    {key:<26}{champ:>10.4f}{chall:>12.4f}{chall - champ:>+10.4f}")
    lines += [
        "",
        f"    rule          promote if {decision.metric} improves by more than "
        f"{decision.margin:.4f}",
        f"    improvement   {decision.improvement:+.4f}",
        f"    holdout looks {decision.holdout_spends}",
        f"    verdict       {decision.verdict.upper()}"
        + (
            f", registered as version {decision.challenger_version}"
            if decision.promoted
            else ", the champion keeps the alias"
        ),
        "",
    ]
    return "\n".join(lines)
