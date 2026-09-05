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

    Order matters, and it is: holdout, then val, then the future batch, then train.

    The holdout is drawn first, stratified on churn over the whole population, so it stays a
    fair scoreboard for any model trained later, including one trained on the future batch.

    Val is drawn next, from the same population, and only then is the future batch carved
    out of what remains. That ordering is the point: val has to be drawn from the
    distribution the champion will serve, or it cannot rank candidate models by anything
    that matters. Session 2 found this the hard way. When val was taken from the
    reference pool instead, it contained no fibre-optic customer, so it scored every
    candidate on an easy sub-population and selected a gradient-boosting model whose
    holdout ROC-AUC was 0.63 against the linear model's 0.80. Drawing val from the
    population lifted the correlation between validation and holdout PR-AUC across the
    twelve sweep configs to 0.97. See docs/decisions.md, 2026-09-05.

    Train is what is left after the future batch is removed, so it still contains no
    fibre-optic customer and the champion still meets that slice for the first time in
    Session 3.
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

    remaining, val = train_test_split(
        rest,
        test_size=val_fraction,
        random_state=seed,
        stratify=rest[TARGET_INT],
    )

    future_mask = remaining[column] == equals
    future = remaining[future_mask]
    train = remaining[~future_mask]

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


class HoldoutBudget:
    """A holdout that can be handed out a fixed number of times, and counts.

    Selection is what corrupts a holdout: score every candidate on it, pick the best, and
    the number you report is the maximum of a sample rather than an estimate. The sweep
    therefore selects on val and gets exactly one look at the holdout, for the winner only.
    This class is where "exactly one" stops being a promise and becomes an exception.
    """

    def __init__(self, frame: pd.DataFrame, budget: int = 1) -> None:
        if budget < 1:
            raise SplitError(f"holdout budget must be at least 1, got {budget}")
        self._frame = frame
        self._budget = budget
        self._spends: list[str] = []

    @property
    def spent(self) -> int:
        return len(self._spends)

    @property
    def reasons(self) -> list[str]:
        return list(self._spends)

    @property
    def rows(self) -> int:
        return len(self._frame)

    def spend(self, reason: str) -> pd.DataFrame:
        """Hand out the holdout once, against the budget, recording why."""
        if self.spent >= self._budget:
            raise SplitError(
                f"holdout budget of {self._budget} is already spent on {self._spends}; "
                f"refused a further look for {reason!r}. Score on val instead."
            )
        self._spends.append(reason)
        return self._frame

    def assert_fully_spent(self) -> None:
        """Fail closed if the budget was not used exactly as planned."""
        if self.spent != self._budget:
            raise SplitError(
                f"expected the holdout to be scored {self._budget} time(s), it was scored "
                f"{self.spent}: {self._spends}"
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
