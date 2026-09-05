"""Step 8: serve the registry champion over HTTP, and prove it from the skeleton run."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from . import tracking
from .register import ChampionInfo, champion_info
from .validate import ID_COLUMN, ValidationError, validate


@dataclass(frozen=True)
class LoadedChampion:
    model: Pipeline
    info: ChampionInfo


def load_champion(name: str = "churn", alias: str = "champion") -> LoadedChampion:
    """Load the model the alias points at, with the version and run id that produced it.

    The alias URI is what the registry documents, so it is tried first. The resolved
    version URI is the fallback, which keeps serving working if a future MLflow drops
    alias support in the sklearn loader.
    """
    info = champion_info(name=name, alias=alias)
    try:
        model = mlflow.sklearn.load_model(info.model_uri)
    except Exception:
        model = mlflow.sklearn.load_model(f"models:/{name}/{info.version}")
    return LoadedChampion(model=model, info=info)


class PredictRequest(BaseModel):
    """One customer. The field names are the source column names, unchanged."""

    customerID: str = Field(default="unknown", description="Identifier, not a feature")
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictResponse(BaseModel):
    # model_name and model_version are the registry's words for these fields; keep them.
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    probability: float
    model_name: str
    model_version: str
    run_id: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    model_version: str
    run_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion once at startup and keep the version on the app state.

    A failure here is recorded rather than raised, so /health can answer 503 with the
    reason instead of the process dying with a stack trace.
    """
    tracking.configure()
    app.state.champion = None
    app.state.load_error = None
    try:
        app.state.champion = load_champion()
    except Exception as exc:
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="mlops-loop churn service",
    description="Serves models:/churn@champion from the MLflow registry.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_champion(request_app: FastAPI) -> LoadedChampion:
    champion = getattr(request_app.state, "champion", None)
    if champion is None:
        reason = getattr(request_app.state, "load_error", None) or "no model loaded"
        raise HTTPException(status_code=503, detail=f"champion not available: {reason}")
    return champion


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report the exact model version and run id currently in memory."""
    champion = _require_champion(app)
    return HealthResponse(
        status="ok",
        model_name=champion.info.name,
        model_version=champion.info.version,
        run_id=champion.info.run_id,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Score one customer. The request goes through the training schema first."""
    champion = _require_champion(app)
    frame = pd.DataFrame([{key: str(value) for key, value in payload.model_dump().items()}])
    try:
        validated = validate(frame, with_target=False).frame
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    probability = float(champion.model.predict_proba(validated[_feature_order()])[0, 1])
    return PredictResponse(
        prediction=int(probability >= 0.5),
        probability=probability,
        model_name=champion.info.name,
        model_version=champion.info.version,
        run_id=champion.info.run_id,
    )


def _feature_order() -> list[str]:
    from .features import FEATURE_COLUMNS

    return FEATURE_COLUMNS


def serve_check(holdout: pd.DataFrame, n: int = 5, log: bool = True) -> dict[str, Any]:
    """Load the champion by alias and score n holdout rows.

    This is the skeleton's proof that step 8 works: the model comes back out of the
    registry, not out of the variable it was just fitted into.
    """
    champion = load_champion()
    sample = holdout.head(n)
    probabilities = champion.model.predict_proba(sample[_feature_order()])[:, 1]

    result = {
        "model_name": champion.info.name,
        "model_version": champion.info.version,
        "model_uri": champion.info.model_uri,
        "run_id": champion.info.run_id,
        "rows": [
            {
                "customerID": str(customer_id),
                "probability": float(probability),
                "prediction": int(probability >= 0.5),
            }
            for customer_id, probability in zip(sample[ID_COLUMN], probabilities)
        ],
    }
    if log:
        mlflow.log_dict(result, "serve_check.json")
    return result


def sample_request(frame: pd.DataFrame, index: int = 0) -> dict[str, Any]:
    """Turn one row of a validated frame into a /predict body. Used by the tests.

    numpy scalars are unwrapped, because a JSON body carries plain Python types.
    """
    row = frame.iloc[index]
    body: dict[str, Any] = {}
    for name in PredictRequest.model_fields:
        value = row[name] if name in row else "unknown"
        body[name] = value.item() if hasattr(value, "item") else value
    return body
