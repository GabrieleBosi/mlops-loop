"""Step 1: get the raw CSV, digest it, keep a parquet copy."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from .tracking import bytes_hash
from .validate import EXPECTED_COLUMNS


class IngestError(Exception):
    """Raised when the source is unreachable, empty or not the expected file."""


@dataclass(frozen=True)
class IngestResult:
    frame: pd.DataFrame
    raw_csv: Path
    raw_parquet: Path
    sha256: str
    rows: int


def _read_source(source: str, timeout: int) -> bytes:
    if source.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(source, timeout=timeout) as response:
                if response.status != 200:
                    raise IngestError(f"{source} returned HTTP {response.status}")
                return response.read()
        except urllib.error.URLError as exc:
            raise IngestError(f"cannot fetch {source}: {exc}") from exc
    path = Path(source)
    if not path.exists():
        raise IngestError(f"source not found: {path}")
    return path.read_bytes()


def ingest(source: str, data_dir: Path | str, timeout: int = 60, log: bool = True) -> IngestResult:
    """Read the source once, hash the exact bytes, write raw.csv and raw.parquet.

    The digest is taken over the downloaded bytes, not over a parsed frame, so it changes
    the moment the source file changes. Blanks are preserved as empty strings here;
    validate.py decides what they mean.
    """
    payload = _read_source(source, timeout)
    if not payload.strip():
        raise IngestError(f"{source} is empty")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = data_dir / "raw.csv"
    raw_csv.write_bytes(payload)

    digest = bytes_hash(payload)
    frame = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)

    header = list(frame.columns)
    if header != EXPECTED_COLUMNS:
        raise IngestError(
            f"unexpected header from {source}.\nexpected {EXPECTED_COLUMNS}\ngot      {header}"
        )
    if frame.empty:
        raise IngestError(f"{source} has a header but no rows")

    raw_parquet = data_dir / "raw.parquet"
    frame.to_parquet(raw_parquet, index=False)

    if log:
        mlflow.log_param("dataset_source", source)
        mlflow.log_param("dataset_sha256", digest)
        mlflow.log_metric("raw_rows", float(len(frame)))
        mlflow.log_input(
            mlflow.data.from_pandas(frame, source=source, name="telco-churn-raw", digest=digest[:16]),
            context="raw",
        )

    return IngestResult(
        frame=frame, raw_csv=raw_csv, raw_parquet=raw_parquet, sha256=digest, rows=len(frame)
    )
