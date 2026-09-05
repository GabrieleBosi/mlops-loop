"""Step 3: the split partitions the data and never leaks the holdout."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_loop.split import Splits, SplitError, assert_holdout_unseen, split

RULE = {"column": "InternetService", "equals": "Fiber optic"}


def _split(frame: pd.DataFrame, seed: int = 42) -> Splits:
    return split(frame, holdout_fraction=0.15, val_fraction=0.20, seed=seed, future_rule=RULE)


def test_splits_are_disjoint_and_cover_the_input(splits: Splits, validated) -> None:
    names = list(splits.as_dict())
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not splits.ids(left) & splits.ids(right)

    union = set().union(*(splits.ids(name) for name in names))
    assert union == set(validated.frame.customerID)
    assert sum(len(part) for part in splits.as_dict().values()) == len(validated.frame)


def test_future_batch_is_all_fibre_and_reference_has_none(splits: Splits) -> None:
    assert (splits.future.InternetService == "Fiber optic").all()
    assert not (splits.train.InternetService == "Fiber optic").any()
    assert not (splits.val.InternetService == "Fiber optic").any()


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
            future_rule={"column": "Nope", "equals": "x"},
        )
