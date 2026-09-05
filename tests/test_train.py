"""Step 5: the sweep runs every config, selects on val, and looks at the holdout once."""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from mlops_loop import train
from mlops_loop.split import HoldoutBudget, SplitError

TINY_SWEEP = {
    "selection": {
        "metric": "pr_auc",
        "split": "val",
        "direction": "higher_is_better",
        "tie_breaker": "brier",
    },
    "configs": [
        {
            "name": "lr-tiny",
            "family": "logistic_regression",
            "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 200, "random_state": 42},
        },
        {
            "name": "hgb-tiny",
            "family": "hist_gradient_boosting",
            "params": {
                "learning_rate": 0.1,
                "max_iter": 20,
                "max_leaf_nodes": 7,
                "early_stopping": False,
                "random_state": 42,
            },
        },
    ],
}


def _child(name: str, pr_auc: float, brier: float = 0.1) -> train.ChildResult:
    return train.ChildResult(
        name=name,
        family="logistic_regression",
        params={},
        run_id=f"run-{name}",
        val_metrics={"pr_auc": pr_auc, "brier": brier},
        seconds=0.0,
    )


# --- the pieces that need no MLflow store -------------------------------------------------


def test_unknown_family_stops() -> None:
    with pytest.raises(train.SweepError, match="unknown model family"):
        train.build_pipeline("random_forest", {})


def test_bad_params_stop_with_the_family_named() -> None:
    with pytest.raises(train.SweepError, match="logistic_regression rejected params"):
        train.build_pipeline("logistic_regression", {"not_a_parameter": 1})


def test_duplicate_config_names_stop() -> None:
    sweep = {"configs": [{"name": "a", "family": "logistic_regression"},
                         {"name": "a", "family": "logistic_regression"}]}
    with pytest.raises(train.SweepError, match="duplicate sweep config name"):
        train.load_configs(sweep)


def test_config_naming_an_unknown_family_stops() -> None:
    sweep = {"configs": [{"name": "a", "family": "xgboost"}]}
    with pytest.raises(train.SweepError, match="unknown family"):
        train.load_configs(sweep)


def test_selection_takes_the_best_val_metric() -> None:
    children = [_child("low", 0.40), _child("high", 0.60), _child("mid", 0.50)]
    assert train.select_winner(children, TINY_SWEEP["selection"]).name == "high"


def test_selection_breaks_ties_on_brier_then_order() -> None:
    tied = [_child("first", 0.50, brier=0.20), _child("second", 0.50, brier=0.10)]
    assert train.select_winner(tied, TINY_SWEEP["selection"]).name == "second"

    identical = [_child("first", 0.50, brier=0.10), _child("second", 0.50, brier=0.10)]
    assert train.select_winner(identical, TINY_SWEEP["selection"]).name == "first"


def test_selection_stops_when_the_metric_is_missing() -> None:
    child = train.ChildResult("x", "logistic_regression", {}, "r", {"roc_auc": 0.9}, 0.0)
    with pytest.raises(train.SweepError, match="missing from children"):
        train.select_winner([child], TINY_SWEEP["selection"])


def test_holdout_budget_refuses_a_second_look(splits) -> None:
    budget = HoldoutBudget(splits.holdout, budget=1)
    assert budget.spend("winner").equals(splits.holdout)
    budget.assert_fully_spent()
    with pytest.raises(SplitError, match="already spent"):
        budget.spend("one more peek")


def test_unspent_holdout_budget_is_an_error(splits) -> None:
    with pytest.raises(SplitError, match="it was scored 0"):
        HoldoutBudget(splits.holdout, budget=1).assert_fully_spent()


# --- the sweep end to end against a temporary store ---------------------------------------


@pytest.fixture(scope="module")
def sweep_run(tmp_tracking, sample_csv, tmp_path_factory):
    report_path = tmp_path_factory.mktemp("reports") / "sweep_comparison.md"
    result = train.run(
        sweep_path=_write_tiny_sweep(tmp_path_factory),
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        run_name="sweep-test",
        report_path=report_path,
    )
    return result, report_path


def _write_tiny_sweep(tmp_path_factory) -> str:
    import yaml

    path = tmp_path_factory.mktemp("configs") / "sweep.yaml"
    path.write_text(yaml.safe_dump(TINY_SWEEP), encoding="utf-8")
    return str(path)


def test_every_config_becomes_a_child_of_one_parent(sweep_run) -> None:
    result, _ = sweep_run
    client = MlflowClient()
    parent = client.get_run(result.parent_run_id)

    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.parent_run_id = '{result.parent_run_id}'",
    )
    assert len(children) == len(TINY_SWEEP["configs"]) == len(result.children)
    assert {c.data.tags["config_name"] for c in children} == {"lr-tiny", "hgb-tiny"}


def test_each_child_logs_what_identifies_it(sweep_run) -> None:
    result, _ = sweep_run
    client = MlflowClient()
    for child in result.children:
        run = client.get_run(child.run_id)
        for param in ("config_name", "dataset_sha256", "feature_code_hash", "model_family"):
            assert param in run.data.params, f"{child.name} is missing param {param}"
        for metric in ("roc_auc", "pr_auc", "recall_at_precision_50", "brier"):
            assert f"val_{metric}" in run.data.metrics, f"{child.name} is missing val_{metric}"
        artifacts = {f.path for f in client.list_artifacts(child.run_id)}
        assert "calibration_val.png" in artifacts
        assert "feature_columns.json" in artifacts


def test_the_holdout_is_looked_at_exactly_once_by_the_winner(sweep_run) -> None:
    result, _ = sweep_run
    assert len(result.holdout_spends) == 1
    assert result.winner.name in result.holdout_spends[0]
    assert result.winner.run_id in result.holdout_spends[0]

    client = MlflowClient()
    scored = [
        child.name
        for child in result.children
        if "holdout_pr_auc" in client.get_run(child.run_id).data.metrics
    ]
    assert scored == [result.winner.name]


def test_the_winner_is_the_best_on_val(sweep_run) -> None:
    result, _ = sweep_run
    best = max(result.children, key=lambda child: child.val_metrics["pr_auc"])
    assert result.winner.name == best.name


def test_the_winner_is_registered_and_takes_the_champion_alias(sweep_run) -> None:
    result, _ = sweep_run
    client = MlflowClient()
    champion = client.get_model_version_by_alias(result.model_name, "champion")
    assert str(champion.version) == result.model_version
    assert champion.run_id == result.winner.run_id


def test_the_report_lists_every_run_id(sweep_run) -> None:
    result, report_path = sweep_run
    text = report_path.read_text(encoding="utf-8")
    assert result.parent_run_id in text
    for child in result.children:
        assert child.run_id in text
    assert "What the sweep showed" in text


def test_regenerating_the_report_keeps_hand_written_notes(sweep_run, tmp_path) -> None:
    from mlops_loop import report

    result, _ = sweep_run
    path = tmp_path / "sweep_comparison.md"
    report.write_sweep_comparison(result, path=path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\nA sentence a person wrote.\n", encoding="utf-8"
    )
    report.write_sweep_comparison(result, path=path)
    assert "A sentence a person wrote." in path.read_text(encoding="utf-8")
