"""Step 3: the split partitions the data and never leaks the holdout."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_loop import config
from mlops_loop.split import Splits, SplitError, assert_holdout_unseen, split

RULE = {"column": "InternetService", "equals": "Fiber optic"}
BATCHES = config.drift_config()["future_batches"]


def _split(frame: pd.DataFrame, seed: int = 42) -> Splits:
    return split(frame, holdout_fraction=0.15, val_fraction=0.20, seed=seed,
                 future_batches=BATCHES)


def test_splits_are_disjoint_and_cover_the_input(splits: Splits, validated) -> None:
    parts = splits.parts()
    id_sets = {name: set(part.customerID) for name, part in parts.items()}
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not id_sets[left] & id_sets[right], f"{left} overlaps {right}"
    assert set().union(*id_sets.values()) == set(validated.frame.customerID)
    assert sum(len(part) for part in parts.values()) == len(validated.frame)


def test_the_canonical_four_splits_still_partition_the_input(splits: Splits, validated) -> None:
    names = list(splits.as_dict())
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not splits.ids(left) & splits.ids(right)

    union = set().union(*(splits.ids(name) for name in names))
    assert union == set(validated.frame.customerID)
    assert sum(len(part) for part in splits.as_dict().values()) == len(validated.frame)


def test_future_batch_is_all_fibre_and_train_has_none(splits: Splits) -> None:
    """Train still never meets the future batch, which is what Session 3 depends on."""
    assert (splits.future.InternetService == "Fiber optic").all()
    assert not (splits.train.InternetService == "Fiber optic").any()


def test_named_batches_match_their_rules_and_do_not_overlap(splits: Splits) -> None:
    monthly = splits.batch("fibre-monthly")
    committed = splits.batch("fibre-committed")
    assert (monthly.InternetService == "Fiber optic").all()
    assert (monthly.Contract == "Month-to-month").all()
    assert (committed.InternetService == "Fiber optic").all()
    assert committed.Contract.isin(["One year", "Two year"]).all()
    assert not set(monthly.customerID) & set(committed.customerID)
    assert len(monthly) + len(committed) == len(splits.future)


def test_a_row_goes_to_the_first_batch_that_claims_it(validated) -> None:
    """Overlapping rules must not double-count a customer."""
    overlapping = [
        {"name": "first", "rules": [{"column": "InternetService", "equals": "Fiber optic"}]},
        {"name": "second", "rules": [{"column": "Contract", "equals": "Month-to-month"}]},
    ]
    result = split(validated.frame, 0.15, 0.20, 42, overlapping)
    assert not set(result.batch("first").customerID) & set(result.batch("second").customerID)
    assert (result.batch("second").InternetService != "Fiber optic").all()


def test_unknown_batch_name_stops(splits: Splits) -> None:
    with pytest.raises(SplitError, match="unknown future batch"):
        splits.batch("nope")


def test_a_batch_without_rules_stops(validated) -> None:
    with pytest.raises(SplitError, match="has no rules"):
        split(validated.frame, 0.15, 0.20, 42, [{"name": "empty", "rules": []}])


def test_val_mirrors_the_holdout_population(splits: Splits) -> None:
    """Val is drawn before the future batch, so it looks like the data the model serves.

    Selecting on a val split that excluded fibre-optic customers picked a model whose
    holdout ROC-AUC was 0.63. This assertion is that bug's regression test.
    """
    val_fibre = (splits.val.InternetService == "Fiber optic").mean()
    holdout_fibre = (splits.holdout.InternetService == "Fiber optic").mean()
    assert val_fibre > 0.2
    assert abs(val_fibre - holdout_fibre) < 0.10
    assert abs(splits.val.churn.mean() - splits.holdout.churn.mean()) < 0.05


def test_holdout_keeps_the_population_mix(splits: Splits, validated) -> None:
    """The holdout is drawn before the future batch, so it still contains fibre customers."""
    assert (splits.holdout.InternetService == "Fiber optic").any()
    expected = round(0.15 * len(validated.frame))
    assert abs(len(splits.holdout) - expected) <= 1


def test_same_seed_gives_the_same_ids(validated) -> None:
    first, second = _split(validated.frame), _split(validated.frame)
    for name in first.as_dict():
        assert first.ids(name) == second.ids(name)


def test_different_seed_gives_a_different_holdout(validated) -> None:
    assert _split(validated.frame, seed=42).ids("holdout") != _split(validated.frame, seed=7).ids("holdout")


def test_holdout_leak_is_caught(splits: Splits) -> None:
    leaked = pd.concat([splits.train, splits.holdout.head(3)])
    with pytest.raises(SplitError, match="holdout leak"):
        assert_holdout_unseen(leaked, splits.holdout)
    assert_holdout_unseen(splits.train, splits.holdout) is None


def test_empty_frame_stops(validated) -> None:
    with pytest.raises(SplitError, match="empty frame"):
        _split(validated.frame.head(0))


def test_unknown_future_column_stops(validated) -> None:
    with pytest.raises(SplitError, match="not in the frame"):
        split(
            validated.frame,
            holdout_fraction=0.15,
            val_fraction=0.20,
            seed=42,
            future_batches={"column": "Nope", "equals": "x"},
        )
