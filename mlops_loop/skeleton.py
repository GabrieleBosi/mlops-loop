"""Steps 1 to 8 in one MLflow run. The rough end-to-end pass the whole repo is built on."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from . import config, evaluate, features, ingest, register, serve, split, tracking, validate


@dataclass(frozen=True)
class SkeletonResult:
    run_id: str
    experiment_id: str
    model_name: str
    model_version: str
    metrics: dict[str, float]
    rows: dict[str, int]
    dataset_sha256: str
    feature_code_hash: str
    git_commit: str
    serve_check: dict[str, Any]


@contextmanager
def step(name: str):
    """Tag the run with each step's outcome, so a failure names the step that stopped."""
    started = time.perf_counter()
    mlflow.set_tag(f"step_{name}", "running")
    try:
        yield
    except Exception:
        mlflow.set_tag(f"step_{name}", "failed")
        raise
    mlflow.set_tag(f"step_{name}", "ok")
    mlflow.log_metric(f"seconds_{name}", round(time.perf_counter() - started, 3))


def run(
    config_path: Path | str | None = None,
    drift_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    source: str | None = None,
    run_name: str = "skeleton",
) -> SkeletonResult:
    """Run the eight steps once and return what the run holds.

    source overrides the configured URL so the tests can point at a local fixture.
    """
    cfg = config.skeleton_config(config_path)
    drift = config.drift_config(drift_path)
    root = config.repo_root()
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    future_rule = drift["future_batch"]

    experiment_id = tracking.configure()
    git = tracking.git_info()

    with mlflow.start_run(run_name=run_name) as active:
        run_id = active.info.run_id
        mlflow.set_tags({**git, "phase": "session-1", "pipeline": "skeleton"})
        mlflow.log_artifact(str(root / "configs" / "skeleton.yaml"), artifact_path="configs")
        mlflow.log_artifact(str(root / "configs" / "drift.yaml"), artifact_path="configs")

        with step("ingest"):
            raw = ingest.ingest(source or cfg["data"]["url"], data_dir)

        with step("validate"):
            validated = validate.validate(raw.frame)
            mlflow.log_metric(
                "total_charges_blank_count", float(validated.total_charges_blank_count)
            )

        with step("split"):
            splits = split.split(
                validated.frame,
                holdout_fraction=cfg["split"]["holdout_fraction"],
                val_fraction=cfg["split"]["val_fraction"],
                seed=cfg["split"]["seed"],
                future_rule=future_rule,
            )
            split.log_splits(
                splits,
                seed=cfg["split"]["seed"],
                holdout_fraction=cfg["split"]["holdout_fraction"],
                val_fraction=cfg["split"]["val_fraction"],
                future_rule=future_rule,
                data_dir=data_dir,
            )

        model_params = {k: v for k, v in cfg["model"].items()}

        with step("train"):
            model = train_model(splits, model_params)

        with step("features"):
            features.log_features(model)

        with step("evaluate"):
            metrics: dict[str, float] = {}
            # future is scored, not selected on. Session 1 needs the baseline so Session 3
            # can say how far the champion drifted on a batch it never trained on.
            for name in ("val", "holdout", "future"):
                scored = evaluate.evaluate(model, getattr(splits, name), prefix=name)
                metrics.update({f"{name}_{key}": value for key, value in scored.items()})
            evaluate.calibration_plot(model, splits.holdout, prefix="holdout")

        with step("register"):
            registered = register.register(
                model,
                splits.train,
                name=cfg["registry"]["name"],
                alias=cfg["registry"]["alias"],
            )
            mlflow.set_tags(
                {
                    "model_name": registered.name,
                    "model_version": registered.version,
                    "model_alias": registered.alias,
                }
            )

        with step("serve"):
            checked = serve.serve_check(splits.holdout, n=5)
            if checked["run_id"] != run_id:
                raise register.RegistryError(
                    f"champion run id {checked['run_id']} is not this run {run_id}"
                )

        result = SkeletonResult(
            run_id=run_id,
            experiment_id=str(experiment_id),
            model_name=registered.name,
            model_version=registered.version,
            metrics=metrics,
            rows={name: len(part) for name, part in splits.as_dict().items()},
            dataset_sha256=raw.sha256,
            feature_code_hash=features.feature_code_hash(),
            git_commit=git["git_commit"],
            serve_check=checked,
        )

    return result


def train_model(splits: split.Splits, params: dict[str, Any]):
    """Fit on the reference train split, with the holdout passed in for the leak check."""
    from .train import fit_model

    return fit_model(splits.train, params, holdout=splits.holdout)


def format_result(result: SkeletonResult) -> str:
    """The one-screen summary the CLI prints. Every number here is in the run."""
    lines = [
        "",
        "skeleton complete",
        f"  run id           {result.run_id}",
        f"  experiment       {result.experiment_id}",
        f"  model            {result.model_name} version {result.model_version} (alias champion)",
        f"  git commit       {result.git_commit}",
        f"  dataset sha256   {result.dataset_sha256}",
        f"  feature hash     {result.feature_code_hash}",
        "  rows             "
        + "  ".join(f"{name}={count}" for name, count in result.rows.items()),
        "  metrics",
    ]
    for name in sorted(result.metrics):
        lines.append(f"    {name:<32} {result.metrics[name]:.4f}")
    lines.append(f"  serve check      {len(result.serve_check['rows'])} holdout rows scored")
    lines.append("")
    return "\n".join(lines)
