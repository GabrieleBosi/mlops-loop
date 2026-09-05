"""Step 10: the challenger is trained, compared and promoted only on merit."""

from __future__ import annotations

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from mlops_loop import config, drift, retrain, skeleton
from mlops_loop.split import SplitError


@pytest.fixture(scope="module")
def champion(tmp_tracking, sample_csv):
    """A registered champion for drift to challenge."""
    return skeleton.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        run_name="retrain-fixture",
    )


@pytest.fixture(scope="module")
def drifted(champion, tmp_tracking, sample_csv, tmp_path_factory):
    """One full drift pass over both configured batches."""
    return drift.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        report_dir=str(tmp_path_factory.mktemp("drift-reports")),
    )


def test_champion_config_round_trips_from_its_run(champion) -> None:
    """The challenger is the champion's specification on more data, not a new model."""
    family, params = retrain.champion_config(champion.run_id)
    configured = config.skeleton_config()["model"]
    assert family == "logistic_regression"
    assert params["C"] == configured["C"]
    assert params["max_iter"] == configured["max_iter"]
    assert params["class_weight"] is None
    assert isinstance(params["C"], float)
    assert isinstance(params["max_iter"], int)


def test_a_run_without_model_family_stops(champion) -> None:
    client = MlflowClient()
    empty = client.create_run(client.get_run(champion.run_id).info.experiment_id).info.run_id
    with pytest.raises(retrain.RetrainError, match="no model_family"):
        retrain.champion_config(empty)


def test_fit_challenger_refuses_to_train_on_the_holdout(splits) -> None:
    """The fixed holdout is never trained on, and this is the assertion that says so."""
    leaked = pd.concat([splits.train, splits.holdout.head(5)], ignore_index=True)
    with pytest.raises(SplitError, match="holdout leak"):
        retrain.fit_challenger(
            leaked,
            splits.batch("fibre-monthly"),
            splits.holdout,
            "logistic_regression",
            {"C": 1.0, "max_iter": 200, "random_state": 42},
            log=False,
        )


def test_the_challenger_never_sees_a_holdout_customer(splits) -> None:
    model, fit_frame = retrain.fit_challenger(
        splits.train,
        splits.batch("fibre-monthly"),
        splits.holdout,
        "logistic_regression",
        {"C": 1.0, "max_iter": 200, "random_state": 42},
        log=False,
    )
    assert not set(fit_frame.customerID) & set(splits.holdout.customerID)
    assert len(fit_frame) == len(splits.train) + len(splits.batch("fibre-monthly"))

    from mlops_loop.features import feature_frame

    assert model.predict_proba(feature_frame(splits.holdout)).shape == (len(splits.holdout), 2)


def test_both_batches_produce_a_logged_decision(drifted) -> None:
    assert len(drifted.outcomes) == 2
    assert [outcome.monitor.batch for outcome in drifted.outcomes] == [
        "fibre-monthly",
        "fibre-committed",
    ]
    for outcome in drifted.outcomes:
        assert outcome.monitor.drifted, f"{outcome.monitor.batch} should drift: train has no fibre"
        assert outcome.decision is not None
        assert outcome.decision.verdict in (retrain.PROMOTED, retrain.REJECTED)


def test_each_decision_run_carries_both_run_ids_and_the_margin(drifted) -> None:
    client = MlflowClient()
    for decision in drifted.decisions:
        run = client.get_run(decision.run_id)
        assert run.data.tags["pipeline"] == "promotion_decision"
        assert run.data.tags["champion_run_id"] == decision.champion_run_id
        assert run.data.tags["challenger_run_id"] == decision.challenger_run_id
        assert run.data.tags["verdict"] == decision.verdict
        assert float(run.data.params["promotion_margin"]) == decision.margin
        assert "improvement_pr_auc" in run.data.metrics
        assert {"champion_holdout_pr_auc", "challenger_holdout_pr_auc"} <= set(run.data.metrics)
        assert "promotion_decision.json" in {
            f.path for f in client.list_artifacts(decision.run_id)
        }


def test_the_verdict_follows_the_margin_rule(drifted) -> None:
    """No outcome is engineered: the rule is applied and whatever it says is recorded."""
    for decision in drifted.decisions:
        expected = decision.improvement > decision.margin
        assert decision.promoted is expected
        assert decision.improvement == pytest.approx(
            decision.challenger_metrics[decision.metric]
            - decision.champion_metrics[decision.metric]
        )


def test_the_holdout_is_looked_at_twice_per_decision(drifted) -> None:
    """Once for the champion, once for the challenger, and the budget refuses a third."""
    for decision in drifted.decisions:
        assert len(decision.holdout_spends) == 2
        assert any("champion" in reason for reason in decision.holdout_spends)
        assert any("challenger" in reason for reason in decision.holdout_spends)


def test_a_promotion_registers_a_version_and_a_rejection_does_not(drifted) -> None:
    client = MlflowClient()
    for decision in drifted.decisions:
        if decision.promoted:
            assert decision.challenger_version is not None
            champion = client.get_model_version_by_alias("churn", "champion")
            assert champion.run_id == decision.challenger_run_id or int(
                champion.version
            ) >= int(decision.challenger_version)
        else:
            assert decision.challenger_version is None


def test_the_champion_alias_always_points_somewhere(drifted) -> None:
    champion = MlflowClient().get_model_version_by_alias("churn", "champion")
    assert str(champion.version) == drifted.final_champion_version


def test_the_drift_report_records_the_decision(drifted, tmp_path_factory) -> None:
    from mlops_loop import report

    folder = tmp_path_factory.mktemp("report-check")
    for outcome in drifted.outcomes:
        path = report.write_drift_report(outcome.monitor, outcome.decision, directory=folder)
        text = path.read_text(encoding="utf-8")
        assert outcome.monitor.run_id in text
        assert "Population stability index" in text
        if outcome.decision is not None:
            assert outcome.decision.run_id in text
            assert outcome.decision.challenger_run_id in text
            assert outcome.decision.verdict in text


def test_an_unknown_batch_name_stops(champion, tmp_tracking, sample_csv) -> None:
    from mlops_loop import monitor

    with pytest.raises(monitor.MonitorError, match="unknown batch"):
        drift.run(batches=["not-a-batch"], source=str(sample_csv),
                  data_dir=tmp_tracking["data_dir"])
