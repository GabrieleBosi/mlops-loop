"""The `drift` command: monitor each future batch, and challenge the champion where it drifted.

monitor.py measures and retrain.py decides; this is the small piece that runs them in order
over every configured batch. It lives apart from both because retrain imports monitor, so the
orchestration cannot sit in either without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from . import config, dataset, monitor, register, retrain, serve, tracking


@dataclass
class BatchOutcome:
    monitor: monitor.MonitorResult
    decision: retrain.Decision | None


@dataclass
class DriftResult:
    data_run_id: str
    outcomes: list[BatchOutcome]
    final_champion_version: str

    @property
    def decisions(self) -> list[retrain.Decision]:
        return [outcome.decision for outcome in self.outcomes if outcome.decision is not None]


def run(
    batches: list[str] | None = None,
    config_path: Path | str | None = None,
    drift_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    source: str | None = None,
    report_dir: Path | str | None = None,
) -> DriftResult:
    """Monitor every configured batch in order, retraining and deciding where PSI breached.

    The champion is reloaded before each batch, so a promotion made on the first batch is the
    thing the second batch is measured against. That is the order the real system would see
    them in.
    """
    cfg = config.skeleton_config(config_path)
    drift_cfg = config.drift_config(drift_path)
    registry = cfg["registry"]

    configured = [entry["name"] for entry in drift_cfg["future_batches"]]
    names = batches or configured
    unknown = [name for name in names if name not in configured]
    if unknown:
        raise monitor.MonitorError(
            f"unknown batch(es) {unknown}; configs/drift.yaml defines {configured}"
        )

    tracking.configure()
    git = tracking.git_info()

    # One ingest, validate and split for all batches, logged once in its own run. Ending it
    # before the monitor runs start keeps those at the top level rather than nested under a
    # run that is only about the data.
    with mlflow.start_run(run_name="drift-data") as data_run:
        data_run_id = data_run.info.run_id
        mlflow.set_tags({**git, "phase": "session-3", "pipeline": "drift-data"})
        mlflow.log_artifact(
            str(config.repo_root() / "configs" / "drift.yaml"), artifact_path="configs"
        )
        prepared = dataset.prepare(
            cfg=cfg, drift=drift_cfg, data_dir=data_dir, source=source, log=True
        )

    outcomes: list[BatchOutcome] = []
    for name in names:
        try:
            champion = serve.load_champion(name=registry["name"], alias=registry["alias"])
        except register.RegistryError as exc:
            raise monitor.MonitorError(
                f"no champion to monitor with: {exc}. "
                "Run `python -m mlops_loop train` first."
            ) from exc

        result = monitor.run(
            batch_name=name,
            cfg=cfg,
            drift=drift_cfg,
            prepared=prepared,
            champion=champion,
            report_dir=str(report_dir) if report_dir else None,
        )
        decision = None
        if result.drifted:
            decision = retrain.run(
                result=result,
                splits=prepared.splits,
                champion_model=champion.model,
                cfg=cfg,
                drift=drift_cfg,
            )
            # Rewrite the batch's report now that the decision exists.
            from . import report as report_module

            report_module.write_drift_report(
                result, decision, directory=str(report_dir) if report_dir else None
            )
        outcomes.append(BatchOutcome(monitor=result, decision=decision))

    final = serve.load_champion(name=registry["name"], alias=registry["alias"])
    return DriftResult(
        data_run_id=data_run_id,
        outcomes=outcomes,
        final_champion_version=final.info.version,
    )


def format_result(result: DriftResult) -> str:
    """Everything the CLI prints for a drift run, batch by batch."""
    parts = [f"\ndrift data run  {result.data_run_id}"]
    for outcome in result.outcomes:
        parts.append(monitor.format_result(outcome.monitor))
        if outcome.decision is not None:
            parts.append(retrain.format_decision(outcome.decision))
        else:
            parts.append(
                f"  no PSI breached its threshold for {outcome.monitor.batch}, "
                "so no challenger was trained\n"
            )
    promoted = [d for d in result.decisions if d.promoted]
    parts.append(
        f"drift complete: {len(result.decisions)} promotion decision(s), "
        f"{len(promoted)} promotion(s). "
        f"Champion is now version {result.final_champion_version}.\n"
    )
    return "\n".join(parts)
