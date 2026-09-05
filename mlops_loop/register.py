"""Step 7: log the model, register it, move the champion alias."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from .features import feature_frame


class RegistryError(Exception):
    """Raised when the registry does not hold what the pipeline just put there."""


PREVIOUS_ALIAS = "previous"


@dataclass(frozen=True)
class Registered:
    name: str
    alias: str
    version: str
    run_id: str
    model_id: str
    model_uri: str
    alias_uri: str
    previous_version: str | None = None


@dataclass(frozen=True)
class ChampionInfo:
    name: str
    alias: str
    version: str
    run_id: str
    model_uri: str


def register(
    model: Pipeline,
    example_frame: pd.DataFrame,
    name: str = "churn",
    alias: str = "champion",
    promote: bool = True,
) -> Registered:
    """Log the fitted pipeline as a model version and point the alias at it.

    Session 1 has one model, so promotion is unconditional. Session 2 replaces the
    promote flag with a comparison against the current champion on the fixed holdout.
    The signature is inferred from real rows, so a wrong-shaped request fails at the
    endpoint rather than deep inside sklearn.
    """
    example = feature_frame(example_frame).head(5)
    signature = infer_signature(example, model.predict_proba(example))

    info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=name,
        signature=signature,
        input_example=example,
    )
    version = getattr(info, "registered_model_version", None)
    if version is None:
        raise RegistryError(f"log_model did not return a registered version for '{name}'")

    previous = promote_to_champion(name, alias, str(version)) if promote else None

    return Registered(
        name=name,
        alias=alias,
        version=str(version),
        run_id=str(info.run_id),
        model_id=str(info.model_id),
        model_uri=str(info.model_uri),
        alias_uri=f"models:/{name}@{alias}",
        previous_version=previous,
    )


def promote_to_champion(name: str, alias: str, version: str) -> str | None:
    """Move the champion alias to version, and park the outgoing one under 'previous'.

    Nothing is deleted. The outgoing champion keeps its version number and stays loadable
    as models:/<name>@previous, which is what a rollback needs and what Session 3's
    challenger comparison reads. Returns the version that was demoted, or None if this is
    the first champion or a re-promotion of the same version.
    """
    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(name, alias)
    except Exception:
        current = None

    demoted: str | None = None
    if current is not None and str(current.version) != str(version):
        client.set_registered_model_alias(name, PREVIOUS_ALIAS, str(current.version))
        demoted = str(current.version)

    client.set_registered_model_alias(name, alias, str(version))
    return demoted


def previous_champion(name: str = "churn") -> ChampionInfo | None:
    """The demoted champion, if there is one. None before the second promotion."""
    try:
        return champion_info(name=name, alias=PREVIOUS_ALIAS)
    except RegistryError:
        return None


def champion_info(name: str = "churn", alias: str = "champion") -> ChampionInfo:
    """Resolve the alias to a version and the run that produced it.

    mlflow 3.16 populates ModelVersion.run_id directly; the logged-model lookup is kept as
    a guard so a future version that leaves it empty is reported rather than served with a
    blank provenance field.
    """
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise RegistryError(f"no model '{name}' with alias '{alias}': {exc}") from exc

    run_id = version.run_id or ""
    if not run_id:
        model_id = getattr(version, "model_id", None)
        if model_id:
            run_id = getattr(client.get_logged_model(model_id), "source_run_id", "") or ""
    if not run_id:
        raise RegistryError(
            f"model '{name}' version {version.version} has no run id; provenance is broken"
        )

    return ChampionInfo(
        name=name,
        alias=alias,
        version=str(version.version),
        run_id=str(run_id),
        model_uri=f"models:/{name}@{alias}",
    )
