"""Step 3: holdout, future batch, train and val. Keyed by customerID, asserted disjoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from .validate import ID_COLUMN, TARGET_INT

SPLIT_NAMES = ("train", "val", "holdout", "future")


class SplitError(Exception):
    """Raised when the split does not partition the input. The pipeline stops here."""


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame
    future: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in SPLIT_NAMES}

    def ids(self, name: str) -> set[str]:
        return set(getattr(self, name)[ID_COLUMN])


def split(
    frame: pd.DataFrame,
    holdout_fraction: float,
    val_fraction: float,
    seed: int,
    future_rule: dict[str, str],
) -> Splits:
    """Partition the validated frame into four disjoint sets.

    Order matters. The holdout is drawn first, stratified on churn over the whole
    population, so it stays a fair scoreboard for any model trained later, including one
    trained on the future batch. The future batch is then carved from what is left, which
    is why the reference model never sees a fibre-optic customer.
    """
    if frame.empty:
        raise SplitError("cannot split an empty frame")
    if not 0.0 < holdout_fraction < 1.0:
        raise SplitError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    if not 0.0 < val_fraction < 1.0:
        raise SplitError(f"val_fraction must be in (0, 1), got {val_fraction}")

    column, equals = future_rule["column"], future_rule["equals"]
    if column not in frame.columns:
        raise SplitError(f"future_batch column '{column}' is not in the frame")

    frame = frame.sort_values(ID_COLUMN, kind="mergesort").reset_index(drop=True)

    rest, holdout = train_test_split(
        frame,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=frame[TARGET_INT],
    )

    future_mask = rest[column] == equals
    future = rest[future_mask]
    reference = rest[~future_mask]

    train, val = train_test_split(
        reference,
        test_size=val_fraction,
        random_state=seed,
        stratify=reference[TARGET_INT],
    )

    splits = Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        holdout=holdout.reset_index(drop=True),
        future=future.reset_index(drop=True),
    )
    _assert_partition(splits, frame)
    return splits


def _assert_partition(splits: Splits, frame: pd.DataFrame) -> None:
    """Fail closed: four non-empty, pairwise disjoint sets that cover the input exactly."""
    parts = splits.as_dict()
    for name, part in parts.items():
        if part.empty:
            raise SplitError(f"split '{name}' is empty")

    id_sets = {name: set(part[ID_COLUMN]) for name, part in parts.items()}
    for name, part in parts.items():
        if len(id_sets[name]) != len(part):
            raise SplitError(f"split '{name}' has duplicate customerIDs")

    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise SplitError(
                    f"splits '{left}' and '{right}' share {len(overlap)} customerIDs, "
                    f"for example {sorted(overlap)[:5]}"
                )

    union = set().union(*id_sets.values())
    expected = set(frame[ID_COLUMN])
    if union != expected:
        raise SplitError(
            f"splits cover {len(union)} ids but the input has {len(expected)}; "
            f"missing {len(expected - union)}, unexpected {len(union - expected)}"
        )


def assert_holdout_unseen(fit_frame: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Assert no holdout customer is in the data a model is about to be fit on.

    Called from train.fit_model. The fixed holdout is scored, never trained on, and this
    is where that rule is enforced rather than assumed.
    """
    overlap = set(fit_frame[ID_COLUMN]) & set(holdout[ID_COLUMN])
    if overlap:
        raise SplitError(
            f"holdout leak: {len(overlap)} holdout customerIDs are in the fit data, "
            f"for example {sorted(overlap)[:5]}"
        )


def log_splits(splits: Splits, seed: int, holdout_fraction: float, val_fraction: float,
               future_rule: dict[str, str], data_dir: Path | str) -> Path:
    """Log the split params, row counts, churn rate per split, and the id lists."""
    mlflow.log_params(
        {
            "split_seed": seed,
            "holdout_fraction": holdout_fraction,
            "val_fraction": val_fraction,
            "future_rule": f"{future_rule['column']} == {future_rule['equals']!r}",
        }
    )
    for name, part in splits.as_dict().items():
        mlflow.log_metric(f"rows_{name}", float(len(part)))
        mlflow.log_metric(f"churn_rate_{name}", float(part[TARGET_INT].mean()))

    ids = {name: sorted(part[ID_COLUMN]) for name, part in splits.as_dict().items()}
    mlflow.log_dict(ids, "split_ids.json")

    splits_dir = Path(data_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.as_dict().items():
        part.to_parquet(splits_dir / f"{name}.parquet", index=False)
    return splits_dir
