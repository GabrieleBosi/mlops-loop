"""The one place MLflow is configured. Every other module assumes configure() ran."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from . import config

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "churn"


def configure(
    tracking_uri: str | None = None,
    experiment: str | None = None,
    artifact_root: str | None = None,
) -> str:
    """Point MLflow at the backend store and select the experiment.

    Arguments win over environment variables, which win over the defaults. The
    experiment is created if it does not exist; artifact_root is only honoured at
    creation time, which is how the tests keep artifacts inside tmp_path.
    Returns the experiment id.
    """
    config.load_dotenv()
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    name = experiment or os.environ.get("MLFLOW_EXPERIMENT") or DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is None:
        experiment_id = client.create_experiment(name, artifact_location=artifact_root)
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def git_info(cwd: Path | str | None = None) -> dict[str, str]:
    """Commit hash and dirty flag for the working tree.

    Returns commit "no-git" when git is absent or this is not a repository, so a run
    outside a checkout still records honestly what it knows.
    """
    root = Path(cwd) if cwd is not None else config.repo_root()

    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=root, capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"git_commit": "no-git", "git_dirty": "unknown"}
    porcelain = run(["git", "status", "--porcelain"])
    dirty = "unknown" if porcelain is None else str(bool(porcelain.strip())).lower()
    return {"git_commit": commit, "git_dirty": dirty}


def code_hash(path: Path | str) -> str:
    """sha256 of a source file's bytes. Used for feature_code_hash."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bytes_hash(payload: bytes) -> str:
    """sha256 of raw bytes. Used for the dataset digest."""
    return hashlib.sha256(payload).hexdigest()
