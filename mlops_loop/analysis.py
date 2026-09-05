"""Error analysis: every holdout failure charged to the first component that could have stopped it.

The field guide's rule is to attribute before fixing, and to fix the biggest cell with the
cheapest fix. Attribution here is code rather than judgement, so the tally is reproducible and
arguable: if a rule is wrong, it is wrong in one readable place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from . import config, dataset, evaluate, features, register, report, serve, tracking
from .tracking import step
from .validate import ID_COLUMN, NUMERIC_COLUMNS, TARGET_INT

COMPONENTS = ("split", "features", "model", "threshold")
NEIGHBOURS = 25
UNINFORMATIVE_BAND = 0.05
SAMPLE_SIZE = 20
SEED = 42


class AnalysisError(Exception):
    """Raised when the analysis cannot be produced. Never a partial table."""


@dataclass(frozen=True)
class FailureCase:
    customer_id: str
    truth: int
    probability: float
    kind: str
    component: str
    reason: str


@dataclass
class ErrorAnalysis:
    run_id: str
    model_name: str
    model_version: str
    champion_run_id: str
    rows: int
    errors: int
    false_negatives: int
    false_positives: int
    threshold: float
    val_threshold: float
    sample_size: int
    seed: int
    uninformative_band: float
    trained_on: str
    train_rows: int
    cases: list[FailureCase]
    tally: dict[str, int]
    full_tally: dict[str, int]
    diagnosis: str
    cheapest_fix: str

    @property
    def biggest_cell(self) -> tuple[str, int]:
        return max(self.tally.items(), key=lambda item: item[1])


def training_frame(champion_run_id: str, splits: Any) -> tuple[pd.DataFrame, str]:
    """The rows the champion was actually fitted on, not the rows train happens to hold.

    A challenger promoted by `drift` was fitted on the reference split plus a future batch,
    and its run records that as the `trained_on` param. Attributing its errors against the
    plain train split would charge the split component for a region the model has in fact
    seen, which would be wrong in the direction that flatters the model.
    """
    try:
        params = mlflow.tracking.MlflowClient().get_run(champion_run_id).data.params
    except Exception as exc:
        raise AnalysisError(f"cannot read the champion's run {champion_run_id}: {exc}") from exc

    spec = params.get("trained_on", "train")
    frames: list[pd.DataFrame] = []
    for part in spec.split("+"):
        part = part.strip()
        if part in ("train", "val", "holdout"):
            frames.append(getattr(splits, part))
        elif part in splits.batches:
            frames.append(splits.batch(part))
        else:
            raise AnalysisError(
                f"champion run {champion_run_id} says it was trained on {spec!r}, and {part!r} "
                "is not a split or a configured batch"
            )
    return pd.concat(frames, ignore_index=True), spec


def best_f1_threshold(model: Pipeline, frame: pd.DataFrame) -> float:
    """The threshold that maximises F1 on a split.

    Chosen on val, never on the holdout: a threshold tuned on the holdout would make the
    "threshold" component of the tally unfalsifiable, because every failure would look
    fixable by a cut nobody could have known in advance.
    """
    truth = frame[TARGET_INT].to_numpy()
    scores = evaluate.predict_proba(model, frame)
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # precision_recall_curve returns one more precision/recall point than thresholds.
    best = int(np.argmax(f1[:-1])) if thresholds.size else 0
    return float(thresholds[best]) if thresholds.size else 0.5


def attribute(
    holdout: pd.DataFrame,
    train_frame: pd.DataFrame,
    model: Pipeline,
    threshold: float,
    val_threshold: float,
    neighbours: int = NEIGHBOURS,
    band: float = UNINFORMATIVE_BAND,
) -> pd.DataFrame:
    """Charge every misclassified holdout row to one component.

    Pipeline order decides precedence: a row the split could not have covered is charged to
    the split even if the model would also have got it wrong, because fixing the model cannot
    help a region the training data never contained.
    """
    scores = evaluate.predict_proba(model, holdout)
    truth = holdout[TARGET_INT].to_numpy()
    predicted = (scores >= threshold).astype(int)
    wrong = predicted != truth
    if not wrong.any():
        raise AnalysisError("the champion makes no mistakes on the holdout, which cannot be right")

    # 1. split: outside the region the training data covers.
    train_categories = {
        column: set(train_frame[column].unique())
        for column in features.FEATURE_COLUMNS
        if column not in NUMERIC_COLUMNS
    }
    ranges = {
        column: (train_frame[column].min(), train_frame[column].max())
        for column in NUMERIC_COLUMNS
    }
    unseen_category = pd.Series(False, index=holdout.index)
    reasons_split: list[str] = []
    for position, (_, row) in enumerate(holdout.iterrows()):
        missing = [
            f"{column}={row[column]!r} never appears in train"
            for column, allowed in train_categories.items()
            if row[column] not in allowed
        ]
        outside = [
            f"{column}={row[column]:.2f} is outside the train range "
            f"{ranges[column][0]:.2f} to {ranges[column][1]:.2f}"
            for column in NUMERIC_COLUMNS
            if row[column] < ranges[column][0] or row[column] > ranges[column][1]
        ]
        both = missing + outside
        unseen_category.iloc[position] = bool(both)
        reasons_split.append("; ".join(both[:2]))

    # 2. features: the neighbourhood carries no signal.
    preprocessor = model.named_steps["preprocessor"]
    train_matrix = preprocessor.transform(features.feature_frame(train_frame))
    holdout_matrix = preprocessor.transform(features.feature_frame(holdout))
    knn = NearestNeighbors(n_neighbors=min(neighbours, len(train_frame))).fit(train_matrix)
    _, indices = knn.kneighbors(holdout_matrix)
    train_target = train_frame[TARGET_INT].to_numpy()
    neighbour_rate = train_target[indices].mean(axis=1)
    base_rate = float(train_target.mean())
    uninformative = np.abs(neighbour_rate - base_rate) < band

    # 4. threshold: right at the validation-optimal cut, wrong at this one.
    at_val_threshold = (scores >= val_threshold).astype(int)
    fixed_by_threshold = wrong & (at_val_threshold == truth)

    components: list[str] = []
    reasons: list[str] = []
    for position in range(len(holdout)):
        if not wrong[position]:
            components.append("")
            reasons.append("")
            continue
        if unseen_category.iloc[position]:
            components.append("split")
            reasons.append(reasons_split[position])
        elif uninformative[position]:
            components.append("features")
            reasons.append(
                f"{neighbours} nearest train rows churn at {neighbour_rate[position]:.2f}, "
                f"against a base rate of {base_rate:.2f}"
            )
        elif fixed_by_threshold[position]:
            components.append("threshold")
            reasons.append(
                f"p={scores[position]:.3f} is on the right side of {val_threshold:.3f} "
                f"and the wrong side of {threshold:.3f}"
            )
        else:
            components.append("model")
            reasons.append(
                f"neighbourhood churns at {neighbour_rate[position]:.2f} but the model scored "
                f"p={scores[position]:.3f}"
            )

    return pd.DataFrame(
        {
            ID_COLUMN: holdout[ID_COLUMN].to_numpy(),
            "truth": truth,
            "probability": scores,
            "predicted": predicted,
            "wrong": wrong,
            "component": components,
            "reason": reasons,
        }
    )


def _diagnosis(tally: dict[str, int], sample_size: int, context: dict[str, Any]) -> str:
    """One paragraph naming what the biggest cell actually means, from the numbers."""
    biggest, count = max(tally.items(), key=lambda item: item[1])
    share = count / sample_size
    texts = {
        "split": (
            f"{count} of {sample_size} sampled failures ({share:.0%}) are rows the training "
            f"data never covered. The champion was fitted on `{context['trained_on']}`, and "
            f"{context['fibre_share']:.0%} of the holdout is fibre-optic, so it is "
            "extrapolating into regions it has no evidence about. This is not a modelling "
            "problem: no estimator can learn a region absent from its training data."
        ),
        "features": (
            f"{count} of {sample_size} sampled failures ({share:.0%}) sit in neighbourhoods "
            "where the feature set carries no signal. The 19 columns describe the contract and "
            "the bill and say nothing about service quality, competitor pricing or support "
            "history, so these customers are indistinguishable from ones who stayed."
        ),
        "model": (
            f"{count} of {sample_size} sampled failures ({share:.0%}) are rows in distribution "
            "whose neighbourhoods do carry signal, and the model still ranked them wrongly. "
            "They are not near misses. Of the "
            f"{context['model_errors']} such failures across the whole holdout, "
            f"{context['confident_fp']} are false positives at a median predicted probability "
            f"of {context['confident_fp_median']:.3f}: the model says these customers are "
            "leaving and they stay. No threshold reaches them, because moving the cut far "
            "enough to catch them would surrender most of the true positives too. "
            "The shape of it is over-prediction. The champion flags "
            f"{context['predicted_positive_rate']:.1%} of the holdout as churners against an "
            f"actual rate of {context['actual_rate']:.1%}, which is what "
            f"`class_weight={context['class_weight']!r}` does to a logistic regression: it "
            "buys recall by shifting every probability up, and PR-AUC, the metric that "
            "selected it, cannot see the cost because it only reads the ranking. Session 2 saw "
            "the same thing from the other end, when this model cleared the gate's Brier line "
            "by 0.0019."
        ),
        "threshold": (
            f"{count} of {sample_size} sampled failures ({share:.0%}) are ranked correctly and "
            f"only cut wrongly: they are right at the validation-optimal threshold "
            f"{context['val_threshold']:.3f} and wrong at the default "
            f"{context['threshold']:.3f}. The model is fine; the decision rule is not."
        ),
    }
    return texts[biggest]


def _cheapest_fix(tally: dict[str, int], context: dict[str, Any]) -> str:
    """The cheapest fix for the biggest cell, named but deliberately not applied."""
    biggest, _ = max(tally.items(), key=lambda item: item[1])
    fixes = {
        "split": (
            "Stop training on a sub-population and serving the whole one. The champion is "
            "fitted on the reference split only, which excludes every fibre-optic customer, "
            "because Session 1 carved that slice out to create a drift story. The cheapest fix "
            "is to fold the promoted challenger's training data back into the default training "
            "set, so the model that serves the population is fitted on the population. That is "
            "a change to `dataset.prepare` and `configs/skeleton.yaml`, not a new model: no new "
            "features, no new dependency, no sweep. The drift machinery keeps working because a "
            "future batch can be defined on any feature, and this repo has several unused."
        ),
        "features": (
            "Add the features the errors are asking for before touching the estimator. The "
            "cheapest version is interaction terms the linear model cannot express on its own, "
            "starting with tenure by contract type and monthly charge by internet service, "
            "which is a change to `features.build_preprocessor` and one sweep to measure it. "
            "Anything that needs data the file does not contain is a different project."
        ),
        "model": (
            "Put calibration into the selection rule, and let the sweep decide again. The "
            "cheapest version is two lines of config and no new code: add Brier to "
            "`configs/sweep.yaml` as a constraint alongside the PR-AUC objective, so a config "
            "that ranks a hair better and calibrates much worse cannot win, and re-run "
            "`python -m mlops_loop train`. The sweep already contains the unweighted twin of "
            "the current champion, so the alternative is measured, not hypothetical: in "
            "Session 2 `lr-c0.05` scored val PR-AUC 0.5731 against the champion's 0.5851 and "
            "val Brier 0.1604 against 0.1868. That is 0.012 of ranking traded for 0.026 of "
            "calibration. "
            "It costs 26 seconds of compute and it is falsifiable: if the tally after the "
            "change still puts most failures in `model`, the diagnosis was wrong and the next "
            "thing to try is capacity or features, in that order. What is not cheap, and not "
            "justified by this table, is adding a calibration layer or a new model family "
            "before the free experiment has been run."
        ),
        "threshold": (
            "Choose the operating threshold instead of inheriting 0.5. Fit it on the validation "
            "split against the cost the business actually carries, log it as a model parameter, "
            "and have `serve` read it rather than hardcoding the comparison. That is a few lines "
            "in `evaluate` and `serve` and no retraining at all."
        ),
    }
    return fixes[biggest]


def run(
    cfg: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    prepared: dataset.Prepared | None = None,
    data_dir: str | None = None,
    source: str | None = None,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
    report_path: str | None = None,
) -> ErrorAnalysis:
    """Score the champion on the holdout, attribute every failure, write the report."""
    cfg = cfg if cfg is not None else config.skeleton_config()
    drift = drift if drift is not None else config.drift_config()
    registry = cfg["registry"]

    try:
        champion = serve.load_champion(name=registry["name"], alias=registry["alias"])
    except register.RegistryError as exc:
        raise AnalysisError(
            f"no champion to analyse: {exc}. Run `python -m mlops_loop train` first."
        ) from exc

    git = tracking.git_info()
    with mlflow.start_run(run_name="error_analysis") as active:
        mlflow.set_tags(
            {
                **git,
                "phase": "session-3",
                "pipeline": "error_analysis",
                "model_name": champion.info.name,
                "model_version": champion.info.version,
                "champion_run_id": champion.info.run_id,
            }
        )
        if prepared is None:
            prepared = dataset.prepare(cfg=cfg, drift=drift, data_dir=data_dir, source=source)
        holdout = prepared.splits.holdout
        train_frame, trained_on = training_frame(champion.info.run_id, prepared.splits)

        with step("attribute"):
            val_threshold = best_f1_threshold(champion.model, prepared.splits.val)
            table = attribute(
                holdout, train_frame, champion.model, 0.5, val_threshold
            )

        failures = table[table["wrong"]].reset_index(drop=True)
        full_tally = {
            component: int((failures["component"] == component).sum())
            for component in COMPONENTS
        }
        sample = failures.sample(
            n=min(sample_size, len(failures)), random_state=seed
        ).sort_values("probability", ascending=False)
        tally = {
            component: int((sample["component"] == component).sum()) for component in COMPONENTS
        }

        cases = [
            FailureCase(
                customer_id=str(row[ID_COLUMN]),
                truth=int(row["truth"]),
                probability=float(row["probability"]),
                kind="false negative" if row["truth"] == 1 else "false positive",
                component=str(row["component"]),
                reason=str(row["reason"]),
            )
            for _, row in sample.iterrows()
        ]

        scores = evaluate.predict_proba(champion.model, holdout)
        model_errors = failures[failures["component"] == "model"]
        confident_fp = model_errors[model_errors["truth"] == 0]
        champion_params = MlflowClient().get_run(champion.info.run_id).data.params
        context = {
            "threshold": 0.5,
            "val_threshold": val_threshold,
            "fibre_share": float((holdout["InternetService"] == "Fiber optic").mean()),
            "trained_on": trained_on,
            "model_errors": len(model_errors),
            "confident_fp": len(confident_fp),
            "confident_fp_median": float(confident_fp["probability"].median())
            if len(confident_fp)
            else float("nan"),
            "predicted_positive_rate": float((scores >= 0.5).mean()),
            "actual_rate": float(holdout[TARGET_INT].mean()),
            "class_weight": champion_params.get("model_class_weight", "unknown"),
        }
        analysis = ErrorAnalysis(
            run_id=active.info.run_id,
            model_name=champion.info.name,
            model_version=champion.info.version,
            champion_run_id=champion.info.run_id,
            rows=len(holdout),
            errors=len(failures),
            false_negatives=int((failures["truth"] == 1).sum()),
            false_positives=int((failures["truth"] == 0).sum()),
            threshold=0.5,
            val_threshold=val_threshold,
            sample_size=len(sample),
            seed=seed,
            uninformative_band=UNINFORMATIVE_BAND,
            trained_on=trained_on,
            train_rows=len(train_frame),
            cases=cases,
            tally=tally,
            full_tally=full_tally,
            diagnosis=_diagnosis(tally, len(sample), context),
            cheapest_fix=_cheapest_fix(tally, context),
        )

        mlflow.log_params(
            {
                "champion_trained_on": trained_on,
                "champion_train_rows": len(train_frame),
                "threshold": 0.5,
                "val_threshold": round(val_threshold, 6),
                "sample_size": len(sample),
                "sample_seed": seed,
                "knn_neighbours": NEIGHBOURS,
                "uninformative_band": UNINFORMATIVE_BAND,
            }
        )
        mlflow.log_metric("holdout_errors", float(len(failures)))
        mlflow.log_metric("holdout_false_negatives", float(analysis.false_negatives))
        mlflow.log_metric("holdout_false_positives", float(analysis.false_positives))
        for component, count in full_tally.items():
            mlflow.log_metric(f"errors_{component}", float(count))
        for component, count in tally.items():
            mlflow.log_metric(f"sampled_errors_{component}", float(count))
        mlflow.log_dict(
            {
                "tally_sampled": tally,
                "tally_all": full_tally,
                "biggest_cell": analysis.biggest_cell[0],
                "cases": [case.__dict__ for case in cases],
            },
            "error_analysis.json",
        )
        written = report.write_error_analysis(analysis, path=report_path)
        mlflow.log_artifact(str(written), artifact_path="reports")

    return analysis


def format_result(analysis: ErrorAnalysis) -> str:
    """The tally the CLI prints."""
    biggest, count = analysis.biggest_cell
    lines = [
        "",
        f"error analysis: {analysis.model_name} version {analysis.model_version} "
        f"on {analysis.rows:,} holdout rows at threshold {analysis.threshold:.2f}",
        f"  analysis run  {analysis.run_id}",
        f"  {analysis.errors} wrong ({analysis.errors / analysis.rows:.1%}): "
        f"{analysis.false_negatives} false negatives, {analysis.false_positives} false positives",
        "",
        f"  {'component':<14}{'sampled':>9}{'all':>7}",
    ]
    for component in COMPONENTS:
        lines.append(
            f"  {component:<14}{analysis.tally[component]:>9}{analysis.full_tally[component]:>7}"
        )
    lines += [
        "",
        f"  biggest cell  {biggest} ({count} of {analysis.sample_size} sampled)",
        "",
    ]
    return "\n".join(lines)
