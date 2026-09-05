"""Step 5: fit one pipeline, and run the explicit sweep over configs/sweep.yaml."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config, dataset, evaluate, features, register, report, tracking
from .features import build_preprocessor, feature_frame
from .split import HoldoutBudget, Splits, assert_holdout_unseen
from .tracking import step
from .validate import TARGET_INT

# The two families the sweep is allowed to build. A config naming anything else stops the
# run rather than silently falling back to a default estimator.
FAMILIES: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


class SweepError(Exception):
    """Raised when the sweep config or the selection cannot be trusted."""


@dataclass(frozen=True)
class SweepConfig:
    name: str
    family: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ChildResult:
    name: str
    family: str
    params: dict[str, Any]
    run_id: str
    val_metrics: dict[str, float]
    seconds: float


@dataclass
class SweepResult:
    parent_run_id: str
    experiment_id: str
    children: list[ChildResult]
    winner: ChildResult
    holdout_metrics: dict[str, float]
    model_name: str
    model_version: str
    previous_version: str | None
    promotion: dict[str, Any]
    dataset_sha256: str
    feature_code_hash: str
    git_commit: str
    rows: dict[str, int]
    holdout_spends: list[str] = field(default_factory=list)
    seconds: float = 0.0


def build_pipeline(family: str, params: dict[str, Any]) -> Pipeline:
    """Preprocessor plus one estimator from a named family, as a single estimator.

    Both families share the preprocessor. The scaler is redundant for the trees and
    harmless, and one shared 45-column matrix is what makes the families comparable: every
    child in the sweep carries the same feature-code hash and the same column list.
    """
    if family not in FAMILIES:
        raise SweepError(
            f"unknown model family {family!r}; configs/sweep.yaml may use {sorted(FAMILIES)}"
        )
    try:
        model = FAMILIES[family](**params)
    except TypeError as exc:
        raise SweepError(f"{family} rejected params {params}: {exc}") from exc
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def fit_model(
    train_frame: pd.DataFrame,
    params: dict[str, Any],
    holdout: pd.DataFrame | None = None,
    family: str = "logistic_regression",
    log: bool = True,
) -> Pipeline:
    """Fit on train_frame after checking the holdout is not in it.

    holdout is passed in so the leak check runs against real ids rather than a promise.
    Comparing ids is not scoring, so this check does not spend the holdout budget.
    """
    if holdout is not None:
        assert_holdout_unseen(train_frame, holdout)

    pipeline = build_pipeline(family, params)
    pipeline.fit(feature_frame(train_frame), train_frame[TARGET_INT])

    if log:
        mlflow.log_params({f"model_{key}": value for key, value in params.items()})
        mlflow.log_param("model_class", type(pipeline.named_steps["model"]).__name__)
        mlflow.log_param("model_family", family)
        mlflow.log_metric("rows_fit", float(len(train_frame)))
    return pipeline


def load_configs(sweep: dict[str, Any]) -> list[SweepConfig]:
    """Read the explicit grid. No expansion happens here or anywhere else."""
    entries = sweep.get("configs")
    if not entries:
        raise SweepError("configs/sweep.yaml has no configs")

    configs: list[SweepConfig] = []
    seen: set[str] = set()
    for entry in entries:
        name, family = entry.get("name"), entry.get("family")
        if not name or not family:
            raise SweepError(f"every sweep config needs a name and a family: {entry}")
        if name in seen:
            raise SweepError(f"duplicate sweep config name {name!r}")
        if family not in FAMILIES:
            raise SweepError(f"config {name!r} names unknown family {family!r}")
        seen.add(name)
        configs.append(
            SweepConfig(name=name, family=family, params=dict(entry.get("params") or {}))
        )
    return configs


def select_winner(children: list[ChildResult], selection: dict[str, Any]) -> ChildResult:
    """Pick the best child on the selection metric, deterministically.

    Selection reads the validation split only. The holdout does not appear in this function
    and cannot influence which config wins.
    """
    metric = selection["metric"]
    tie_breaker = selection.get("tie_breaker", "brier")
    higher_is_better = selection.get("direction", "higher_is_better") == "higher_is_better"

    missing = [child.name for child in children if metric not in child.val_metrics]
    if missing:
        raise SweepError(f"selection metric {metric!r} missing from children {missing}")

    order = {child.name: index for index, child in enumerate(children)}

    def key(child: ChildResult) -> tuple[float, float, int]:
        primary = child.val_metrics[metric]
        return (
            -primary if higher_is_better else primary,
            child.val_metrics.get(tie_breaker, 0.0),
            order[child.name],
        )

    return sorted(children, key=key)[0]


def run(
    config_path: Path | str | None = None,
    sweep_path: Path | str | None = None,
    drift_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    source: str | None = None,
    run_name: str = "sweep",
    report_path: Path | str | None = None,
) -> SweepResult:
    """Run every config as a child run, select on val, spend the holdout once, promote.

    One parent run holds the data steps and the outcome. Each child holds one config's
    params, its validation metrics, its calibration plot, and the digests that say which
    data and which feature code produced them.
    """
    started = time.perf_counter()
    cfg = config.skeleton_config(config_path)
    sweep_cfg = config.sweep_config(sweep_path)
    drift = config.drift_config(drift_path)
    root = config.repo_root()
    configs = load_configs(sweep_cfg)
    selection = sweep_cfg["selection"]

    experiment_id = tracking.configure()
    git = tracking.git_info()

    with mlflow.start_run(run_name=run_name) as parent:
        parent_run_id = parent.info.run_id
        mlflow.set_tags({**git, "phase": "session-2", "pipeline": "sweep"})
        for name in ("skeleton.yaml", "sweep.yaml", "drift.yaml"):
            mlflow.log_artifact(str(root / "configs" / name), artifact_path="configs")

        prepared = dataset.prepare(cfg=cfg, drift=drift, data_dir=data_dir, source=source)
        splits = prepared.splits
        budget = HoldoutBudget(splits.holdout, budget=1)

        mlflow.log_params(
            {
                "sweep_configs": len(configs),
                "sweep_families": ",".join(sorted({c.family for c in configs})),
                "selection_metric": selection["metric"],
                "selection_split": selection["split"],
            }
        )

        children: list[ChildResult] = []
        models: dict[str, Pipeline] = {}
        with step("sweep"):
            for entry in configs:
                child, model = _run_child(entry, splits, prepared, parent_run_id)
                children.append(child)
                models[child.name] = model

        with step("select"):
            winner = select_winner(children, selection)
            mlflow.set_tags({"winner_name": winner.name, "winner_run_id": winner.run_id})
            mlflow.log_metric(
                f"winner_val_{selection['metric']}", winner.val_metrics[selection["metric"]]
            )

        with step("holdout"):
            # The only look at the holdout in this command, for the winner only.
            holdout = budget.spend(f"winner {winner.name} ({winner.run_id})")
            model = models[winner.name]
            holdout_metrics = evaluate.evaluate(model, holdout, prefix="holdout", log=False)
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})
            budget.assert_fully_spent()

        with step("register"):
            registered, promotion = _register_winner(
                model=model,
                winner=winner,
                splits=splits,
                holdout=holdout,
                holdout_metrics=holdout_metrics,
                registry=cfg["registry"],
            )
            mlflow.set_tags(
                {
                    "model_name": registered.name,
                    "model_version": registered.version,
                    "model_alias": registered.alias,
                    "previous_version": registered.previous_version or "none",
                }
            )
            mlflow.log_dict(promotion, "promotion.json")

        result = SweepResult(
            parent_run_id=parent_run_id,
            experiment_id=str(experiment_id),
            children=children,
            winner=winner,
            holdout_metrics=holdout_metrics,
            model_name=registered.name,
            model_version=registered.version,
            previous_version=registered.previous_version,
            promotion=promotion,
            dataset_sha256=prepared.dataset_sha256,
            feature_code_hash=features.feature_code_hash(),
            git_commit=git["git_commit"],
            rows={name: len(part) for name, part in splits.as_dict().items()},
            holdout_spends=budget.reasons,
            seconds=round(time.perf_counter() - started, 1),
        )

        with step("report"):
            written = report.write_sweep_comparison(result, path=report_path)
            mlflow.log_artifact(str(written), artifact_path="reports")

    return result


def _run_child(
    entry: SweepConfig,
    splits: Splits,
    prepared: dataset.Prepared,
    parent_run_id: str,
) -> tuple[ChildResult, Pipeline]:
    """One config: fit on train, score on val, log everything that identifies the run."""
    started = time.perf_counter()
    with mlflow.start_run(run_name=entry.name, nested=True) as child:
        mlflow.set_tags(
            {
                "config_name": entry.name,
                "parent_run_id": parent_run_id,
                "phase": "session-2",
                "pipeline": "sweep-child",
            }
        )
        mlflow.log_params(
            {"config_name": entry.name, "dataset_sha256": prepared.dataset_sha256}
        )
        model = fit_model(splits.train, entry.params, holdout=splits.holdout, family=entry.family)
        features.log_features(model)

        val_metrics = evaluate.evaluate(model, splits.val, prefix="val")
        evaluate.calibration_plot(model, splits.val, prefix="val")

        child_result = ChildResult(
            name=entry.name,
            family=entry.family,
            params=entry.params,
            run_id=child.info.run_id,
            val_metrics=val_metrics,
            seconds=round(time.perf_counter() - started, 2),
        )
        mlflow.log_metric("seconds_config", child_result.seconds)
    return child_result, model


def _register_winner(
    model: Pipeline,
    winner: ChildResult,
    splits: Splits,
    holdout: pd.DataFrame,
    holdout_metrics: dict[str, float],
    registry: dict[str, Any],
) -> tuple[register.Registered, dict[str, Any]]:
    """Log the winner's model into its own child run, register it, move the aliases.

    The model artifact belongs to the run that trained it, so the winning child run is
    resumed rather than logging the model against the parent. Only the winner's model is
    stored: the sweep is seeded and finishes in minutes, so keeping a pickle per config to
    avoid a re-fit is the wrong trade, and each child already carries the params, the
    dataset digest and the feature-code hash that reproduce it.
    """
    name, alias = registry["name"], registry["alias"]
    outgoing = _current_champion(name, alias)
    outgoing_metrics = _recorded_holdout_metrics(outgoing.run_id) if outgoing else {}

    with mlflow.start_run(run_id=winner.run_id, nested=True):
        mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})
        evaluate.calibration_plot(model, holdout, prefix="holdout")
        registered = register.register(model, splits.train, name=name, alias=alias, promote=True)

    promotion = {
        "decision": "promoted",
        "rule": (
            "Session 2 promotes the sweep winner selected by validation PR-AUC. The "
            "head-to-head rule that refuses a challenger which does not beat the champion on "
            "the fixed holdout arrives in Session 3. Today the eval gate is what stops a "
            "regression going anywhere."
        ),
        "incoming": {
            "version": registered.version,
            "run_id": winner.run_id,
            "config_name": winner.name,
            "family": winner.family,
            "holdout": holdout_metrics,
        },
        "outgoing": (
            {
                "version": outgoing.version,
                "run_id": outgoing.run_id,
                "alias_now": register.PREVIOUS_ALIAS,
                # Read back from the outgoing champion's own run. Re-scoring it here would
                # be a second look at the holdout.
                "holdout_recorded": outgoing_metrics,
            }
            if outgoing
            else None
        ),
    }
    return registered, promotion


def _current_champion(name: str, alias: str) -> register.ChampionInfo | None:
    try:
        return register.champion_info(name=name, alias=alias)
    except register.RegistryError:
        return None


def _recorded_holdout_metrics(run_id: str) -> dict[str, float]:
    """Holdout metrics already logged on a run. No new scoring, so no new holdout look."""
    try:
        data = MlflowClient().get_run(run_id).data.metrics
    except Exception:
        return {}
    return {key[len("holdout_") :]: value for key, value in data.items() if key.startswith("holdout_")}


def format_result(result: SweepResult) -> str:
    """The one-screen summary the CLI prints. Every number here is in a run."""
    lines = [
        "",
        "sweep complete",
        f"  parent run       {result.parent_run_id}",
        f"  configs          {len(result.children)} in {result.seconds}s",
        f"  git commit       {result.git_commit}",
        f"  dataset sha256   {result.dataset_sha256}",
        f"  feature hash     {result.feature_code_hash}",
        "",
        f"  {'config':<28}{'family':<22}{'val pr_auc':>11}{'val roc_auc':>12}{'val brier':>11}",
    ]
    for child in sorted(result.children, key=lambda c: -c.val_metrics["pr_auc"]):
        mark = "  <-- champion" if child.name == result.winner.name else ""
        lines.append(
            f"  {child.name:<28}{child.family:<22}"
            f"{child.val_metrics['pr_auc']:>11.4f}{child.val_metrics['roc_auc']:>12.4f}"
            f"{child.val_metrics['brier']:>11.4f}{mark}"
        )
    lines += [
        "",
        f"  winner           {result.winner.name} ({result.winner.run_id})",
        f"  registered       {result.model_name} version {result.model_version}, alias champion",
        f"  previous         {result.previous_version or 'none, this is the first champion'}",
        f"  holdout looks    {result.holdout_spends}",
        "  holdout metrics",
    ]
    for key in sorted(result.holdout_metrics):
        lines.append(f"    {key:<26} {result.holdout_metrics[key]:.4f}")
    lines.append("")
    return "\n".join(lines)
