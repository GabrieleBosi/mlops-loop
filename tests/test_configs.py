"""The committed configs are engineering artifacts. These are their contract."""

from __future__ import annotations

import pytest

from mlops_loop import config, gate, train


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


@pytest.mark.parametrize("name", ["skeleton.yaml", "sweep.yaml", "thresholds.yaml", "drift.yaml"])
def test_configs_load(name: str) -> None:
    loader = {
        "skeleton.yaml": config.skeleton_config,
        "sweep.yaml": config.sweep_config,
        "thresholds.yaml": config.thresholds_config,
        "drift.yaml": config.drift_config,
    }[name]
    assert loader()
