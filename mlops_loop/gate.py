"""Step 11: the gate. Loads the champion, scores the holdout, exits non-zero below threshold.

The gate is deliberately dumb. It does not retrain, it does not choose, and it cannot pass
by being clever: it reads the thresholds from a file that was committed before the model
existed, scores whatever currently holds the champion alias, and compares. Anything it
cannot compute is a failure, not a skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from . import config, dataset, evaluate, register, serve, tracking
from .tracking import step

HIGHER = "higher_is_better"
LOWER = "lower_is_better"


class GateError(Exception):
    """Raised when the gate cannot be evaluated at all. Never a silent pass."""


@dataclass(frozen=True)
class Check:
    metric: str
    direction: str
    threshold: float
    value: float

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold if self.direction == HIGHER else self.value <= self.threshold

    @property
    def margin(self) -> float:
        """How much room is left. Negative when the check failed."""
        return self.value - self.threshold if self.direction == HIGHER else self.threshold - self.value


@dataclass(frozen=True)
class GateResult:
    checks: list[Check]
    model_name: str
    model_version: str
    champion_run_id: str
    run_id: str
    split: str
    rows: int

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]


def load_thresholds(path: Path | str | None = None) -> dict[str, Any]:
    """Read configs/thresholds.yaml and check every line is usable before anything runs."""
    cfg = config.thresholds_config(path)
    for metric, spec in cfg["metrics"].items():
        if spec.get("direction") not in (HIGHER, LOWER):
            raise GateError(
                f"threshold {metric!r} has direction {spec.get('direction')!r}; "
                f"expected {HIGHER!r} or {LOWER!r}"
            )
        if not isinstance(spec.get("threshold"), (int, float)):
            raise GateError(f"threshold {metric!r} has no numeric threshold")
    return cfg


def run(
    config_path: Path | str | None = None,
    thresholds_path: Path | str | None = None,
    drift_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    source: str | None = None,
    run_name: str = "eval",
) -> GateResult:
    """Score the champion on the gate's split and compare against the thresholds.

    The gate scores the holdout on purpose. That is not in tension with the sweep's
    one-look budget: the sweep may not use the holdout to choose between candidates, while
    the gate exists to measure the one model that was already chosen.
    """
    cfg = config.skeleton_config(config_path)
    drift = config.drift_config(drift_path)
    thresholds = load_thresholds(thresholds_path)
    split_name = thresholds.get("split", "holdout")

    tracking.configure()
    git = tracking.git_info()

    registry = cfg["registry"]
    try:
        champion = serve.load_champion(name=registry["name"], alias=registry["alias"])
    except register.RegistryError as exc:
        raise GateError(
            f"no model {registry['name']!r} with alias {registry['alias']!r} in the registry "
            f"at {mlflow.get_tracking_uri()}. Run `python -m mlops_loop train` first. ({exc})"
        ) from exc

    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags(
            {
                **git,
                "phase": "session-2",
                "pipeline": "gate",
                "model_name": champion.info.name,
                "model_version": champion.info.version,
                "champion_run_id": champion.info.run_id,
            }
        )
        mlflow.log_artifact(
            str(config.repo_root() / "configs" / "thresholds.yaml"), artifact_path="configs"
        )

        prepared = dataset.prepare(cfg=cfg, drift=drift, data_dir=data_dir, source=source)
        frame = getattr(prepared.splits, split_name)

        with step("score"):
            scored = evaluate.evaluate(champion.model, frame, prefix=split_name, log=True)

        checks: list[Check] = []
        for metric, spec in thresholds["metrics"].items():
            key = metric[len(f"{split_name}_") :] if metric.startswith(f"{split_name}_") else metric
            if key not in scored:
                raise GateError(
                    f"threshold names {metric!r} but the {split_name} split produced "
                    f"{sorted(scored)}. Fix the threshold file or the metric, do not skip it."
                )
            checks.append(
                Check(
                    metric=metric,
                    direction=spec["direction"],
                    threshold=float(spec["threshold"]),
                    value=float(scored[key]),
                )
            )

        result = GateResult(
            checks=checks,
            model_name=champion.info.name,
            model_version=champion.info.version,
            champion_run_id=champion.info.run_id,
            run_id=active.info.run_id,
            split=split_name,
            rows=len(frame),
        )

        mlflow.log_metrics({f"gate_margin_{check.metric}": check.margin for check in checks})
        mlflow.log_metric("gate_failures", float(len(result.failures)))
        mlflow.set_tag("gate_result", "pass" if result.passed else "fail")
        mlflow.log_dict(
            {
                "passed": result.passed,
                "model": {"name": result.model_name, "version": result.model_version,
                          "run_id": result.champion_run_id},
                "split": split_name,
                "rows": result.rows,
                "checks": [
                    {
                        "metric": check.metric,
                        "direction": check.direction,
                        "threshold": check.threshold,
                        "value": check.value,
                        "margin": check.margin,
                        "passed": check.passed,
                    }
                    for check in checks
                ],
            },
            "gate.json",
        )

    return result


def format_result(result: GateResult) -> str:
    """The table the CLI prints. Reads the same whether it passed or failed."""
    lines = [
        "",
        f"eval gate: {result.model_name} version {result.model_version} "
        f"on {result.rows:,} {result.split} rows",
        f"  champion run  {result.champion_run_id}",
        f"  gate run      {result.run_id}",
        "",
        f"  {'metric':<34}{'value':>10}{'threshold':>12}{'margin':>10}   result",
    ]
    for check in result.checks:
        arrow = ">=" if check.direction == HIGHER else "<="
        lines.append(
            f"  {check.metric:<34}{check.value:>10.4f}{arrow + f'{check.threshold:.4f}':>12}"
            f"{check.margin:>+10.4f}   {'pass' if check.passed else 'FAIL'}"
        )
    lines.append("")
    if result.passed:
        lines.append(f"  gate passed: {len(result.checks)} of {len(result.checks)} checks")
    else:
        failed = ", ".join(check.metric for check in result.failures)
        lines.append(f"  gate FAILED on {len(result.failures)} check(s): {failed}")
    lines.append("")
    return "\n".join(lines)
