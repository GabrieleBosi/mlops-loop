"""Step 11: the gate passes, fails, and refuses to run rather than passing by accident."""

from __future__ import annotations

import pytest
import yaml
from mlflow.tracking import MlflowClient

from mlops_loop import gate, skeleton

LENIENT = {
    "split": "holdout",
    "metrics": {
        "holdout_roc_auc": {"direction": "higher_is_better", "threshold": 0.50},
        "holdout_brier": {"direction": "lower_is_better", "threshold": 0.90},
    },
}

IMPOSSIBLE = {
    "split": "holdout",
    "metrics": {
        "holdout_roc_auc": {"direction": "higher_is_better", "threshold": 0.999},
        "holdout_brier": {"direction": "lower_is_better", "threshold": 0.001},
    },
}


def _thresholds(tmp_path, payload) -> str:
    path = tmp_path / "thresholds.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def champion(tmp_tracking, sample_csv):
    """One registered champion for the gate to score. The skeleton is the cheapest way."""
    return skeleton.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        run_name="gate-fixture",
    )


def test_gate_passes_and_records_the_margin(champion, tmp_tracking, sample_csv, tmp_path) -> None:
    result = gate.run(
        thresholds_path=_thresholds(tmp_path, LENIENT),
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
    )
    assert result.passed
    assert not result.failures
    assert result.model_version == champion.model_version
    assert result.champion_run_id == champion.run_id
    assert result.rows == champion.rows["holdout"]
    assert all(check.margin > 0 for check in result.checks)

    run = MlflowClient().get_run(result.run_id)
    assert run.data.tags["gate_result"] == "pass"
    assert run.data.metrics["gate_failures"] == 0.0
    assert {f.path for f in MlflowClient().list_artifacts(result.run_id)} >= {"gate.json"}


def test_gate_fails_when_the_threshold_is_out_of_reach(
    champion, tmp_tracking, sample_csv, tmp_path
) -> None:
    result = gate.run(
        thresholds_path=_thresholds(tmp_path, IMPOSSIBLE),
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
    )
    assert not result.passed
    assert {check.metric for check in result.failures} == set(IMPOSSIBLE["metrics"])
    assert all(check.margin < 0 for check in result.failures)

    run = MlflowClient().get_run(result.run_id)
    assert run.data.tags["gate_result"] == "fail"
    assert run.data.metrics["gate_failures"] == 2.0


def test_the_printed_table_names_the_failing_metrics(
    champion, tmp_tracking, sample_csv, tmp_path
) -> None:
    result = gate.run(
        thresholds_path=_thresholds(tmp_path, IMPOSSIBLE),
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
    )
    text = gate.format_result(result)
    assert "FAIL" in text
    assert "holdout_roc_auc" in text
    assert f"version {result.model_version}" in text


def test_a_threshold_naming_an_unknown_metric_stops(
    champion, tmp_tracking, sample_csv, tmp_path
) -> None:
    """A metric the pipeline does not produce is a failure, never a silently skipped line."""
    payload = {
        "split": "holdout",
        "metrics": {"holdout_f1": {"direction": "higher_is_better", "threshold": 0.5}},
    }
    with pytest.raises(gate.GateError, match="do not skip it"):
        gate.run(
            thresholds_path=_thresholds(tmp_path, payload),
            source=str(sample_csv),
            data_dir=tmp_tracking["data_dir"],
        )


def test_a_bad_direction_stops_before_anything_runs(tmp_path) -> None:
    payload = {
        "split": "holdout",
        "metrics": {"holdout_roc_auc": {"direction": "bigger", "threshold": 0.5}},
    }
    with pytest.raises(gate.GateError, match="expected"):
        gate.load_thresholds(_thresholds(tmp_path, payload))


def test_an_empty_threshold_file_stops(tmp_path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text(yaml.safe_dump({"split": "holdout", "metrics": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty mapping"):
        gate.load_thresholds(str(path))


def test_gate_without_a_champion_stops(tmp_tracking, sample_csv, tmp_path, monkeypatch) -> None:
    """No registered champion is a failure with an instruction, not a crash."""
    import mlflow

    from mlops_loop import tracking

    empty = tmp_path / "empty.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{empty.as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "gate-without-champion")
    mlflow.set_tracking_uri(f"sqlite:///{empty.as_posix()}")
    tracking.configure()

    with pytest.raises(gate.GateError, match="python -m mlops_loop train"):
        gate.run(
            thresholds_path=_thresholds(tmp_path, LENIENT),
            source=str(sample_csv),
            data_dir=tmp_tracking["data_dir"],
        )

    mlflow.set_tracking_uri(tmp_tracking["uri"])
