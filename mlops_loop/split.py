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
DEFAULT_BATCH_NAME = "future"


class SplitError(Exception):
    """Raised when the split does not partition the input. The pipeline stops here."""


@dataclass(frozen=True)
class Splits:
    """The four canonical splits, with the future pool broken into named batches.

    `future` is the union of `batches` and exists so every command that only cares about
    "data the champion has never trained on" keeps working. The partition assertions run
    over the batches individually, because that is what actually has to be disjoint.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame
    batches: dict[str, pd.DataFrame]

    @property
    def future(self) -> pd.DataFrame:
        return pd.concat(list(self.batches.values()), ignore_index=True)

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in SPLIT_NAMES}

    def parts(self) -> dict[str, pd.DataFrame]:
        """The real partition: the three fixed splits plus one entry per future batch."""
        parts = {name: getattr(self, name) for name in ("train", "val", "holdout")}
        parts.update({f"batch:{name}": frame for name, frame in self.batches.items()})
        return parts

    def batch(self, name: str) -> pd.DataFrame:
        if name not in self.batches:
            raise SplitError(f"unknown future batch {name!r}; configured: {sorted(self.batches)}")
        return self.batches[name]

    def ids(self, name: str) -> set[str]:
        return set(getattr(self, name)[ID_COLUMN])


def split(
    frame: pd.DataFrame,
    holdout_fraction: float,
    val_fraction: float,
    seed: int,
    future_batches: list[dict] | dict[str, str],
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

    Train is what is left after the future batches are removed, so it still contains no
    fibre-optic customer and the champion still meets that slice for the first time in
    Session 3.

    future_batches is the list from configs/drift.yaml: each entry has a name and a list of
    rules, all of which a row must match. Batches are carved in order and a row goes to the
    first batch that claims it, so two overlapping rules cannot double-count a customer. A
    single {column, equals} mapping is accepted as one batch named "future".
    """
    if frame.empty:
        raise SplitError("cannot split an empty frame")
    if not 0.0 < holdout_fraction < 1.0:
        raise SplitError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    if not 0.0 < val_fraction < 1.0:
        raise SplitError(f"val_fraction must be in (0, 1), got {val_fraction}")

    batches_spec = normalise_batches(future_batches)
    for spec in batches_spec:
        for rule in spec["rules"]:
            if rule["column"] not in frame.columns:
                raise SplitError(
                    f"future batch {spec['name']!r} names column {rule['column']!r}, "
                    f"which is not in the frame"
                )

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

    batches: dict[str, pd.DataFrame] = {}
    unclaimed = pd.Series(True, index=remaining.index)
    for spec in batches_spec:
        mask = unclaimed & batch_mask(remaining, spec["rules"])
        batches[spec["name"]] = remaining[mask].reset_index(drop=True)
        unclaimed &= ~mask
    train = remaining[unclaimed]

    splits = Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        holdout=holdout.reset_index(drop=True),
        batches=batches,
    )
    _assert_partition(splits, frame)
    return splits


def normalise_batches(future_batches: list[dict] | dict[str, str]) -> list[dict]:
    """Accept either the configs/drift.yaml list or one plain {column, equals} rule."""
    if isinstance(future_batches, dict):
        future_batches = [{"name": DEFAULT_BATCH_NAME, "rules": [dict(future_batches)]}]
    if not future_batches:
        raise SplitError("at least one future batch must be configured")

    normalised: list[dict] = []
    seen: set[str] = set()
    for entry in future_batches:
        name = entry.get("name")
        if not name:
            raise SplitError(f"every future batch needs a name: {entry}")
        if name in seen:
            raise SplitError(f"duplicate future batch name {name!r}")
        seen.add(name)
        rules = entry.get("rules") or ([entry] if "column" in entry else [])
        if not rules:
            raise SplitError(f"future batch {name!r} has no rules")
        for rule in rules:
            if "column" not in rule or not ({"equals", "in"} & set(rule)):
                raise SplitError(
                    f"future batch {name!r} rule {rule} needs a column and either "
                    f"'equals' or 'in'"
                )
        normalised.append({"name": name, "rules": rules})
    return normalised


def batch_mask(frame: pd.DataFrame, rules: list[dict]) -> pd.Series:
    """Rows matching every rule. 'equals' is one value, 'in' is a list of them."""
    mask = pd.Series(True, index=frame.index)
    for rule in rules:
        column = rule["column"]
        if "equals" in rule:
            mask &= frame[column] == rule["equals"]
        else:
            mask &= frame[column].isin(list(rule["in"]))
    return mask


def _assert_partition(splits: Splits, frame: pd.DataFrame) -> None:
    """Fail closed: non-empty, pairwise disjoint sets that cover the input exactly."""
    parts = splits.parts()
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
               future_batches: list[dict] | dict[str, str], data_dir: Path | str) -> Path:
    """Log the split params, row counts, churn rate per split and batch, and the id lists."""
    specs = normalise_batches(future_batches)
    mlflow.log_params(
        {
            "split_seed": seed,
            "holdout_fraction": holdout_fraction,
            "val_fraction": val_fraction,
            "future_batches": ",".join(spec["name"] for spec in specs),
            "future_rule": "; ".join(
                spec["name"] + ": " + " and ".join(
                    f"{rule['column']} {'==' if 'equals' in rule else 'in'} "
                    f"{rule.get('equals', rule.get('in'))!r}"
                    for rule in spec["rules"]
                )
                for spec in specs
            ),
        }
    )
    for name, part in splits.as_dict().items():
        mlflow.log_metric(f"rows_{name}", float(len(part)))
        mlflow.log_metric(f"churn_rate_{name}", float(part[TARGET_INT].mean()))
    for name, part in splits.batches.items():
        mlflow.log_metric(f"rows_batch_{name}", float(len(part)))
        mlflow.log_metric(f"churn_rate_batch_{name}", float(part[TARGET_INT].mean()))

    ids = {name: sorted(part[ID_COLUMN]) for name, part in splits.parts().items()}
    mlflow.log_dict(ids, "split_ids.json")

    splits_dir = Path(data_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.as_dict().items():
        part.to_parquet(splits_dir / f"{name}.parquet", index=False)
    for name, part in splits.batches.items():
        part.to_parquet(splits_dir / f"batch_{name}.parquet", index=False)
    return splits_dir
