"""Steps 1 to 8 end to end on the 500-row fixture, against a temporary MLflow store."""

from __future__ import annotations

import re

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from mlops_loop import serve, skeleton


@pytest.fixture(scope="module")
def skeleton_run(tmp_tracking, sample_csv):
    """One skeleton run on the fixture. Everything below reads the run it produced."""
    return skeleton.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        run_name="skeleton-test",
    )


def test_skeleton_run_holds_what_the_brief_requires(skeleton_run, tmp_tracking) -> None:
    client = MlflowClient()
    run = client.get_run(skeleton_run.run_id)

    assert run.info.status == "FINISHED"
    assert run.info.experiment_id == tmp_tracking["experiment_id"]

    for param in ("dataset_sha256", "dataset_source", "feature_code_hash", "split_seed",
                  "holdout_fraction", "val_fraction", "future_rule", "model_C",
                  "model_solver", "model_max_iter"):
        assert param in run.data.params, f"missing param {param}"
    assert re.fullmatch(r"[0-9a-f]{64}", run.data.params["dataset_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", run.data.params["feature_code_hash"])

    for prefix in ("val", "holdout", "future"):
        for metric in ("roc_auc", "pr_auc", "recall_at_precision_50", "brier"):
            assert f"{prefix}_{metric}" in run.data.metrics, f"missing metric {prefix}_{metric}"

    assert run.data.metrics["total_charges_blank_count"] == 11.0
    assert run.data.metrics["raw_rows"] == 500.0
    for name in ("train", "val", "holdout", "future"):
        assert run.data.metrics[f"rows_{name}"] > 0

    assert "git_commit" in run.data.tags
    assert run.data.tags["phase"] == "session-1"
    for stepname in ("ingest", "validate", "split", "features", "train", "evaluate",
                     "register", "serve"):
        assert run.data.tags[f"step_{stepname}"] == "ok"

    artifacts = {f.path for f in client.list_artifacts(skeleton_run.run_id)}
    assert {"calibration_holdout.png", "split_ids.json", "feature_columns.json",
            "serve_check.json"} <= artifacts

    inputs = run.inputs.dataset_inputs
    assert inputs and inputs[0].dataset.name == "telco-churn-raw"


def test_metrics_are_in_range(skeleton_run) -> None:
    for name, value in skeleton_run.metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"
    assert skeleton_run.metrics["holdout_roc_auc"] > 0.5


def test_splits_do_not_overlap_in_the_logged_ids(skeleton_run) -> None:
    ids = mlflow.artifacts.load_dict(
        f"{MlflowClient().get_run(skeleton_run.run_id).info.artifact_uri}/split_ids.json"
    )
    assert set(ids) == {"train", "val", "holdout", "batch:fibre-monthly", "batch:fibre-committed"}
    seen: set[str] = set()
    for name, values in ids.items():
        assert not seen & set(values), f"{name} overlaps an earlier split"
        seen |= set(values)
    assert len(seen) == 500


def test_champion_alias_points_at_this_run(skeleton_run) -> None:
    client = MlflowClient()
    version = client.get_model_version_by_alias("churn", "champion")
    assert str(version.version) == skeleton_run.model_version
    assert version.run_id == skeleton_run.run_id


def test_serve_check_scored_five_holdout_rows(skeleton_run) -> None:
    check = skeleton_run.serve_check
    assert check["model_uri"] == "models:/churn@champion"
    assert check["run_id"] == skeleton_run.run_id
    assert len(check["rows"]) == 5
    for row in check["rows"]:
        assert 0.0 <= row["probability"] <= 1.0
        assert row["prediction"] in (0, 1)


def test_endpoint_serves_the_registered_champion(skeleton_run, tmp_tracking, validated) -> None:
    with TestClient(serve.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_name": "churn",
            "model_version": skeleton_run.model_version,
            "run_id": skeleton_run.run_id,
        }

        body = serve.sample_request(validated.frame)
        response = client.post("/predict", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["prediction"] in (0, 1)
        assert 0.0 <= payload["probability"] <= 1.0
        assert payload["model_version"] == skeleton_run.model_version
        assert payload["run_id"] == skeleton_run.run_id


def test_endpoint_rejects_an_unknown_category(skeleton_run, tmp_tracking, validated) -> None:
    with TestClient(serve.app) as client:
        body = serve.sample_request(validated.frame)
        body["Contract"] = "Three year"
        response = client.post("/predict", json=body)
        assert response.status_code == 422
        assert "Contract" in response.text
