"""Shared fixtures. Every test runs offline against a temporary MLflow store."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "telco_sample_500.csv"


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    """The committed 500-row sample. No network, identical on every machine."""
    return FIXTURE


@pytest.fixture()
def sample_raw(sample_csv: Path) -> pd.DataFrame:
    """The fixture read the way ingest reads it: strings, blanks kept as empty strings."""
    return pd.read_csv(sample_csv, dtype=str, keep_default_na=False)


@pytest.fixture()
def validated(sample_raw: pd.DataFrame):
    from mlops_loop.validate import validate

    return validate(sample_raw)


@pytest.fixture()
def splits(validated):
    from mlops_loop.split import split

    return split(
        validated.frame,
        holdout_fraction=0.15,
        val_fraction=0.20,
        seed=42,
        future_rule={"column": "InternetService", "equals": "Fiber optic"},
    )


@pytest.fixture(scope="module")
def tmp_tracking(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Point MLflow at a store inside a temporary directory, with a fresh experiment name.

    Module scoped: the integration test fits a model, and one store per module keeps the
    suite to a single skeleton run. The real mlflow.db and mlruns/ are never touched.
    """
    tmp_path = tmp_path_factory.mktemp("tracking")
    store = tmp_path / "mlflow.db"
    artifacts = tmp_path / "mlruns"
    artifacts.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{store.as_posix()}"
    experiment = f"test-{uuid.uuid4().hex[:8]}"

    import mlflow

    from mlops_loop import tracking

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", uri)
        patch.setenv("MLFLOW_EXPERIMENT", experiment)
        mlflow.set_tracking_uri(uri)
        experiment_id = tracking.configure(
            tracking_uri=uri, experiment=experiment, artifact_root=artifacts.as_uri()
        )
        yield {
            "uri": uri,
            "experiment": experiment,
            "experiment_id": experiment_id,
            "data_dir": str(tmp_path / "data"),
        }
        mlflow.end_run()
