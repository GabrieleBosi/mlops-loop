"""Step 2: schema validation. Owns the column and category sets the encoder reuses."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

# Captured from the full source file on 2026-09-05 (7043 rows). The schema checks against
# these and features.py builds the one-hot encoder from the same dict, so the validated
# categories and the encoded columns cannot drift apart.
CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

CATEGORICAL_COLUMNS: list[str] = list(CATEGORIES)
NUMERIC_COLUMNS: list[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_INT = "churn"

EXPECTED_COLUMNS: list[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


class ValidationError(Exception):
    """Raised when the data does not match the contract. The pipeline stops here."""


@dataclass(frozen=True)
class ValidateResult:
    frame: pd.DataFrame
    total_charges_blank_count: int


def build_schema(with_target: bool = True) -> pa.DataFrameSchema:
    """The contract for a validated frame.

    strict=True so an unexpected column stops the run. tenure has no upper bound: a future
    batch with longer-tenured customers is new data, not a violation.
    """
    columns: dict[str, pa.Column] = {
        ID_COLUMN: pa.Column(str, nullable=False, unique=True),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "tenure": pa.Column(int, pa.Check.ge(0), nullable=False),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "TotalCharges": pa.Column(float, pa.Check.ge(0.0), nullable=False),
    }
    for column, values in CATEGORIES.items():
        columns[column] = pa.Column(str, pa.Check.isin(values), nullable=False)
    if with_target:
        columns[TARGET_COLUMN] = pa.Column(str, pa.Check.isin(["No", "Yes"]), nullable=False)
        columns[TARGET_INT] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(columns, strict=True, ordered=False, coerce=False)


def _coerce_numeric(frame: pd.DataFrame, column: str, as_int: bool) -> pd.Series:
    values = pd.to_numeric(frame[column].astype(str).str.strip(), errors="coerce")
    bad = values.isna()
    if bad.any():
        examples = frame.loc[bad, [ID_COLUMN, column]].head(5).to_dict("records")
        raise ValidationError(
            f"{column}: {int(bad.sum())} value(s) are not numeric, for example {examples}"
        )
    return values.astype("int64") if as_int else values.astype("float64")


def validate(raw: pd.DataFrame, with_target: bool = True) -> ValidateResult:
    """Coerce the raw string frame to the contract, or stop with the reason.

    The 11 blank TotalCharges in the source file are handled before the schema runs: every
    blank must be a tenure-0 customer, in which case the true total billed is 0.0. A blank
    on any other row means the source changed and the run stops.
    """
    frame = raw.copy()

    expected = EXPECTED_COLUMNS if with_target else [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]
    missing = [c for c in expected if c not in frame.columns]
    extra = [c for c in frame.columns if c not in expected]
    if missing or extra:
        raise ValidationError(f"column mismatch: missing={missing} unexpected={extra}")

    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()

    tenure = _coerce_numeric(frame, "tenure", as_int=True)
    blank_mask = frame["TotalCharges"] == ""
    blank_count = int(blank_mask.sum())
    if blank_count:
        offenders = frame.loc[blank_mask & (tenure != 0), ID_COLUMN].tolist()
        if offenders:
            raise ValidationError(
                f"TotalCharges is blank on {len(offenders)} row(s) with tenure != 0, "
                f"for example {offenders[:5]}. The source changed; stopping."
            )
        frame.loc[blank_mask, "TotalCharges"] = "0.0"

    frame["tenure"] = tenure
    frame["SeniorCitizen"] = _coerce_numeric(frame, "SeniorCitizen", as_int=True)
    frame["MonthlyCharges"] = _coerce_numeric(frame, "MonthlyCharges", as_int=False)
    frame["TotalCharges"] = _coerce_numeric(frame, "TotalCharges", as_int=False)

    if with_target:
        if not frame[TARGET_COLUMN].isin(["No", "Yes"]).all():
            bad = sorted(set(frame[TARGET_COLUMN]) - {"No", "Yes"})
            raise ValidationError(f"{TARGET_COLUMN} has unexpected values: {bad}")
        frame[TARGET_INT] = (frame[TARGET_COLUMN] == "Yes").astype("int64")

    schema = build_schema(with_target=with_target)
    try:
        frame = schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        raise ValidationError(f"schema violations:\n{exc.failure_cases.to_string()}") from exc

    return ValidateResult(frame=frame, total_charges_blank_count=blank_count)
