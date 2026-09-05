"""Step 2: the schema is the contract, and it fails closed."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_loop.validate import ValidationError, validate


def test_blank_total_charges_counted_and_zeroed(sample_raw: pd.DataFrame) -> None:
    result = validate(sample_raw)
    assert result.total_charges_blank_count == 11

    blank_ids = sample_raw.loc[sample_raw.TotalCharges.str.strip() == "", "customerID"]
    zeroed = result.frame[result.frame.customerID.isin(blank_ids)]
    assert len(zeroed) == 11
    assert (zeroed.TotalCharges == 0.0).all()
    assert (zeroed.tenure == 0).all()


def test_blank_total_charges_with_tenure_stops(sample_raw: pd.DataFrame) -> None:
    frame = sample_raw.copy()
    row = frame.index[frame.TotalCharges.str.strip() != ""][0]
    frame.loc[row, "TotalCharges"] = ""

    with pytest.raises(ValidationError, match="tenure != 0"):
        validate(frame)


def test_unexpected_category_stops(sample_raw: pd.DataFrame) -> None:
    frame = sample_raw.copy()
    frame.loc[frame.index[0], "Contract"] = "Three year"

    with pytest.raises(ValidationError, match="schema violations"):
        validate(frame)


def test_unexpected_column_stops(sample_raw: pd.DataFrame) -> None:
    frame = sample_raw.copy()
    frame["LifetimeValue"] = "100"

    with pytest.raises(ValidationError, match="unexpected="):
        validate(frame)


def test_non_numeric_total_charges_stops(sample_raw: pd.DataFrame) -> None:
    frame = sample_raw.copy()
    frame.loc[frame.index[0], "MonthlyCharges"] = "n/a"

    with pytest.raises(ValidationError, match="not numeric"):
        validate(frame)


def test_dtypes_and_target(sample_raw: pd.DataFrame) -> None:
    frame = validate(sample_raw).frame
    assert frame.tenure.dtype == "int64"
    assert frame.SeniorCitizen.dtype == "int64"
    assert frame.MonthlyCharges.dtype == "float64"
    assert frame.TotalCharges.dtype == "float64"
    assert frame.churn.dtype == "int64"
    assert set(frame.churn.unique()) == {0, 1}
    assert (frame.churn == (frame.Churn == "Yes").astype(int)).all()


def test_without_target(sample_raw: pd.DataFrame) -> None:
    """A prediction request has no Churn column and must still validate."""
    frame = sample_raw.drop(columns=["Churn"]).head(3)
    result = validate(frame, with_target=False)
    assert "churn" not in result.frame.columns
    assert len(result.frame) == 3
