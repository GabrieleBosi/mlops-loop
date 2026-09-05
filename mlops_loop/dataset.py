"""Steps 1 to 3 as one call, so skeleton, train and eval build identical splits.

The splits are a pure function of the source bytes, the fractions and the seed. Rebuilding
them in each command is cheaper than shipping them around, and it means the eval gate scores
the same 1,057 customers the sweep's winner was measured on without trusting a file on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from . import config, ingest, split, validate
from .tracking import step


@dataclass(frozen=True)
class Prepared:
    raw: pd.DataFrame
    frame: pd.DataFrame
    splits: split.Splits
    dataset_sha256: str
    rows: int
    total_charges_blank_count: int


def prepare(
    cfg: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    data_dir: Path | str | None = None,
    source: str | None = None,
    log: bool = True,
) -> Prepared:
    """Ingest, validate and split, logging each step to the active run when log is True."""
    cfg = cfg if cfg is not None else config.skeleton_config()
    drift = drift if drift is not None else config.drift_config()
    data_dir = Path(data_dir) if data_dir is not None else config.repo_root() / "data"
    future_batches = drift["future_batches"]

    with step("ingest"):
        raw = ingest.ingest(source or cfg["data"]["url"], data_dir, log=log)

    with step("validate"):
        validated = validate.validate(raw.frame)
        if log:
            mlflow.log_metric(
                "total_charges_blank_count", float(validated.total_charges_blank_count)
            )

    with step("split"):
        splits = split.split(
            validated.frame,
            holdout_fraction=cfg["split"]["holdout_fraction"],
            val_fraction=cfg["split"]["val_fraction"],
            seed=cfg["split"]["seed"],
            future_batches=future_batches,
        )
        if log:
            split.log_splits(
                splits,
                seed=cfg["split"]["seed"],
                holdout_fraction=cfg["split"]["holdout_fraction"],
                val_fraction=cfg["split"]["val_fraction"],
                future_batches=future_batches,
                data_dir=data_dir,
            )

    return Prepared(
        raw=raw.frame,
        frame=validated.frame,
        splits=splits,
        dataset_sha256=raw.sha256,
        rows=raw.rows,
        total_charges_blank_count=validated.total_charges_blank_count,
    )
