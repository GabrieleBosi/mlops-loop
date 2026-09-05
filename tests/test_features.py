"""Step 4: the encoded column set is fixed, whatever the fit data happens to contain."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mlops_loop.features import (
    FEATURE_COLUMNS,
    build_preprocessor,
    encoded_columns,
    feature_code_hash,
    feature_frame,
)
from mlops_loop.validate import CATEGORIES


def test_identifier_and_target_are_not_features() -> None:
    assert "customerID" not in FEATURE_COLUMNS
    assert "Churn" not in FEATURE_COLUMNS
    assert "churn" not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 19


def test_column_set_is_the_same_when_a_category_is_missing(splits) -> None:
    """The train split has no fibre-optic customer and must still produce that column."""
    assert not (splits.train.InternetService == "Fiber optic").any()

    on_train = build_preprocessor().fit(feature_frame(splits.train))
    on_all = build_preprocessor().fit(feature_frame(pd.concat([splits.train, splits.future])))

    assert encoded_columns(on_train) == encoded_columns(on_all)
    assert "InternetService_Fiber optic" in encoded_columns(on_train)

    expected = len(FEATURE_COLUMNS) - len(CATEGORIES) + sum(len(v) for v in CATEGORIES.values())
    assert len(encoded_columns(on_train)) == expected


def test_matrix_width_is_the_same_on_every_split(splits) -> None:
    fitted = build_preprocessor().fit(feature_frame(splits.train))
    widths = {name: fitted.transform(feature_frame(part)).shape[1]
              for name, part in splits.as_dict().items()}
    assert len(set(widths.values())) == 1


def test_unknown_category_raises_at_transform(splits) -> None:
    fitted = build_preprocessor().fit(feature_frame(splits.train))
    unseen = feature_frame(splits.holdout).head(1)
    unseen.loc[unseen.index[0], "Contract"] = "Three year"
    with pytest.raises(ValueError, match="unknown categories"):
        fitted.transform(unseen)


def test_feature_code_hash_is_stable_hex() -> None:
    first = feature_code_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first == feature_code_hash()


def test_feature_frame_reports_missing_columns(splits) -> None:
    with pytest.raises(KeyError, match="tenure"):
        feature_frame(splits.train.drop(columns=["tenure"]))


def test_unfitted_preprocessor_does_not_transform(splits) -> None:
    with pytest.raises(NotFittedError):
        build_preprocessor().transform(feature_frame(splits.train))
