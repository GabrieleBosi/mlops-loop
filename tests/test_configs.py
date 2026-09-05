"""The committed configs are engineering artifacts. These are their contract."""

from __future__ import annotations

import pytest

from mlops_loop import config, gate, train
from mlops_loop.validate import EXPECTED_COLUMNS


def test_sweep_covers_at_least_two_families_and_eight_configs() -> None:
    configs = train.load_configs(config.sweep_config())
    assert len(configs) >= 8
    assert len({entry.family for entry in configs}) >= 2
    assert {"logistic_regression", "hist_gradient_boosting"} <= {e.family for e in configs}


def test_every_sweep_config_builds() -> None:
    """A config that cannot be instantiated should fail here, not 20 minutes into a sweep."""
    for entry in train.load_configs(config.sweep_config()):
        train.build_pipeline(entry.family, entry.params)


def test_the_sweep_contains_the_incumbent() -> None:
    """The sweep must be able to reproduce the model it is trying to replace."""
    incumbent = config.skeleton_config()["model"]
    configs = train.load_configs(config.sweep_config())
    matches = [
        entry
        for entry in configs
        if entry.family == "logistic_regression"
        and entry.params.get("C") == incumbent["C"]
        and entry.params.get("class_weight") == incumbent.get("class_weight")
    ]
    assert matches, (
        f"no sweep config reproduces configs/skeleton.yaml's model {incumbent}; "
        "the sweep cannot then be a regression check on the champion"
    )


def test_selection_is_pr_auc_on_val() -> None:
    selection = config.sweep_config()["selection"]
    assert selection["metric"] == "pr_auc"
    assert selection["split"] == "val"


def test_thresholds_cover_all_four_metrics_on_the_holdout() -> None:
    cfg = gate.load_thresholds()
    assert cfg["split"] == "holdout"
    assert set(cfg["metrics"]) == {
        "holdout_roc_auc",
        "holdout_pr_auc",
        "holdout_recall_at_precision_50",
        "holdout_brier",
    }


def test_every_threshold_line_carries_a_comment() -> None:
    """Each threshold has to say where it came from. An uncommented number is a guess."""
    text = (config.repo_root() / "configs" / "thresholds.yaml").read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("  holdout_"):
            continue
        preceding = [
            lines[back].strip()
            for back in range(index - 1, max(index - 6, -1), -1)
            if lines[back].strip().startswith("#")
        ]
        assert preceding, f"threshold {line.strip()} has no comment above it"


@pytest.mark.parametrize(
    "name", ["skeleton.yaml", "sweep.yaml", "thresholds.yaml", "drift.yaml"]
)
def test_configs_load(name: str) -> None:
    loader = {
        "skeleton.yaml": config.skeleton_config,
        "sweep.yaml": config.sweep_config,
        "thresholds.yaml": config.thresholds_config,
        "drift.yaml": config.drift_config,
    }[name]
    assert loader()


def test_drift_defines_two_batches_as_feature_slices() -> None:
    batches = config.drift_config()["future_batches"]
    assert len(batches) >= 2
    for entry in batches:
        assert entry["rules"], f"{entry['name']} has no rules"
        for rule in entry["rules"]:
            assert rule["column"] in EXPECTED_COLUMNS
            assert "equals" in rule or "in" in rule


def test_every_model_feature_has_a_psi_threshold() -> None:
    """An unlisted feature is a feature nobody decided to watch."""
    from mlops_loop.features import FEATURE_COLUMNS

    thresholds = config.drift_config()["psi"]["thresholds"]
    assert set(thresholds) == set(FEATURE_COLUMNS)
    assert all(value > 0 for value in thresholds.values())


def test_the_prediction_distribution_is_watched_more_tightly_than_features() -> None:
    psi_cfg = config.drift_config()["psi"]
    assert psi_cfg["prediction_threshold"] <= min(psi_cfg["thresholds"].values())


def test_the_promotion_margin_is_positive_and_explicit() -> None:
    promotion = config.drift_config()["promotion"]
    assert promotion["margin"] > 0
    assert promotion["metric"] == "pr_auc"
    assert promotion["split"] == "holdout"
