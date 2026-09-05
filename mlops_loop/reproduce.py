"""The `reproduce` command: the whole loop from a clean clone, in one call.

skeleton, train, eval, drift, reports, and then eval once more because drift can change the
champion and a gate that ran before the last promotion has not gated anything. Nothing here is
new work; it is the same commands CI runs, in the order a reader would run them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import analysis, config, drift, gate, skeleton, train


@dataclass
class Stage:
    name: str
    seconds: float
    detail: str


@dataclass
class ReproduceResult:
    stages: list[Stage] = field(default_factory=list)
    skeleton_run_id: str = ""
    sweep_run_id: str = ""
    gate_run_id: str = ""
    final_gate_run_id: str = ""
    analysis_run_id: str = ""
    monitor_run_ids: list[str] = field(default_factory=list)
    decision_run_ids: list[str] = field(default_factory=list)
    final_champion_version: str = ""
    passed: bool = False

    @property
    def seconds(self) -> float:
        return round(sum(stage.seconds for stage in self.stages), 1)


def run(
    config_path: Path | str | None = None,
    source: str | None = None,
    data_dir: Path | str | None = None,
) -> ReproduceResult:
    """Run every command in order and time each one. Any failure stops the whole thing."""
    result = ReproduceResult()

    def stage(name: str):
        started = time.perf_counter()

        def done(detail: str) -> None:
            result.stages.append(
                Stage(name=name, seconds=round(time.perf_counter() - started, 1), detail=detail)
            )

        return done

    done = stage("skeleton")
    baseline = skeleton.run(config_path=config_path, source=source, data_dir=data_dir)
    result.skeleton_run_id = baseline.run_id
    done(f"run {baseline.run_id}, registered version {baseline.model_version}")

    done = stage("train")
    sweep = train.run(config_path=config_path, source=source, data_dir=data_dir)
    result.sweep_run_id = sweep.parent_run_id
    done(
        f"parent {sweep.parent_run_id}, {len(sweep.children)} configs, "
        f"winner {sweep.winner.name} as version {sweep.model_version}"
    )

    done = stage("eval")
    first_gate = gate.run(config_path=config_path, source=source, data_dir=data_dir)
    result.gate_run_id = first_gate.run_id
    if not first_gate.passed:
        raise SystemExit(
            "reproduce stopped: the eval gate failed on the sweep winner.\n"
            + gate.format_result(first_gate)
        )
    done(f"run {first_gate.run_id}, {len(first_gate.checks)} checks passed")

    done = stage("drift")
    drifted = drift.run(config_path=config_path, source=source, data_dir=data_dir)
    result.monitor_run_ids = [outcome.monitor.run_id for outcome in drifted.outcomes]
    result.decision_run_ids = [decision.run_id for decision in drifted.decisions]
    result.final_champion_version = drifted.final_champion_version
    done(
        f"{len(drifted.outcomes)} batch(es), {len(drifted.decisions)} decision(s), "
        f"{len([d for d in drifted.decisions if d.promoted])} promotion(s)"
    )

    done = stage("reports")
    errors = analysis.run(source=source, data_dir=data_dir)
    result.analysis_run_id = errors.run_id
    done(f"run {errors.run_id}, biggest cell {errors.biggest_cell[0]}")

    # The gate runs last because drift can promote a challenger, and a gate that ran before
    # the final promotion has not gated what is actually serving.
    done = stage("eval (final champion)")
    final_gate = gate.run(config_path=config_path, source=source, data_dir=data_dir)
    result.final_gate_run_id = final_gate.run_id
    result.passed = final_gate.passed
    done(
        f"run {final_gate.run_id}, version {final_gate.model_version}, "
        f"{'passed' if final_gate.passed else 'FAILED'}"
    )
    return result


def format_result(result: ReproduceResult) -> str:
    """The timing table the CLI prints."""
    lines = [
        "",
        "reproduce complete" if result.passed else "reproduce finished with a FAILING gate",
        "",
        f"  {'stage':<24}{'seconds':>9}   detail",
    ]
    for stage in result.stages:
        lines.append(f"  {stage.name:<24}{stage.seconds:>9.1f}   {stage.detail}")
    lines += [
        f"  {'total':<24}{result.seconds:>9.1f}",
        "",
        f"  skeleton run     {result.skeleton_run_id}",
        f"  sweep run        {result.sweep_run_id}",
        f"  monitor runs     {', '.join(result.monitor_run_ids)}",
        f"  decision runs    {', '.join(result.decision_run_ids) or 'none'}",
        f"  analysis run     {result.analysis_run_id}",
        f"  gate runs        {result.gate_run_id} then {result.final_gate_run_id}",
        f"  final champion   version {result.final_champion_version}",
        "",
    ]
    return "\n".join(lines)
