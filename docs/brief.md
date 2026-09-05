# mlops-loop: brief

## What this is
One tabular ML system taken through the whole lifecycle on MLflow, then the same tracking discipline applied to a GenAI evaluation suite. Everything a reviewer needs to verify a claim is a logged run.

Two tracks, one method.

Track A, classical, this repo: telco customer churn. Tracked experiments, a model registry with a champion alias, a serving endpoint that loads from the registry, drift monitoring on new batches, a retraining trigger that trains a challenger and promotes it only if it beats the champion on a fixed holdout, and a CI gate that fails closed.

Track B, GenAI, in the order-processing-workflow repo: that repo's evaluation suite logged into MLflow with traces, extended with safety and multilingual cases, with a calibrated LLM judge and acceptance thresholds enforced in CI. This README links to it.

## Why these choices
- Level 2 fixed workflow, no LLM in Track A. Every step is deterministic code. This is the honest architecture for a tabular pipeline and the field guide's default: code is free, exact and testable.
- MLflow: self-hosted, widely named in job descriptions, and one tool covers tracking, registry, GenAI evaluation and tracing.
- Telco churn: public, 7,043 rows, 21 columns, a binary target, and enough categorical structure to produce real drift when sliced.
- Drift is not simulated with noise. The future batch is a slice defined by a feature (fibre-optic customers, or the top tenure quartile), so the shift is real and explainable.

## Data
IBM Telco Customer Churn.
Source: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Verified 2026-09-04: HTTP 200, 7,043 data rows, header `customerID, gender, SeniorCitizen, ..., MonthlyCharges, TotalCharges, Churn`.
Target: Churn (Yes/No).
Known quirk: TotalCharges is a string column with 11 blank values. Coerce to numeric, count the blanks, log the count as a metric, and handle them explicitly. Never let them become silent NaN.

Fallback if the URL dies: UCI "Predict Students' Dropout and Academic Success" (dataset id 697) through the ucimlrepo package. Record the switch in docs/decisions.md.

## Decomposition (written before code, kept current)

| # | Step | Executor | Input -> Output | Risk |
|---|------|----------|-----------------|------|
| 1 | Ingest | code | CSV URL -> data/raw.parquet, sha256 digest logged | source changes silently |
| 2 | Validate | code (pandera) | raw -> validated frame, or stop with the violation | blank TotalCharges, unexpected categories |
| 3 | Split | code | validated -> reference (train, val, holdout) and future batch; split rule and seed logged | leakage across the split |
| 4 | Features | code | frame -> matrix; feature-code hash and column list logged | leakage, encoding drift |
| 5 | Train sweep | code (scikit-learn) | matrix x config grid -> one MLflow child run per config | overfitting the holdout by peeking |
| 6 | Evaluate | code | run -> ROC-AUC, PR-AUC, recall at precision 0.5, Brier score, calibration plot | metric gaming |
| 7 | Register | code | best run by PR-AUC on val -> registry version, alias champion | promoting on noise |
| 8 | Serve | code (FastAPI) | JSON -> prediction, model version, run id | stale model in memory |
| 9 | Monitor | code | new batch -> PSI per feature, PSI on the prediction distribution | thresholds too loose |
| 10 | Retrain trigger | code | PSI above threshold -> challenger trained -> compared on the fixed holdout -> promote or reject, both logged | champion churn |
| 11 | Gate | code (pytest, thresholds.yaml) | runs -> pass or fail, non-zero exit | thresholds set after seeing results |

Executor is code in every row. No step needs an LLM. That is the point.

## Repo layout
```
mlops-loop/
  mlops_loop/              package
    __main__.py            CLI: python -m mlops_loop <command>
    tracking.py            the one place that configures MLflow (uri, experiment, tags)
    ingest.py  validate.py  split.py  features.py  train.py  evaluate.py
    register.py  serve.py  monitor.py  retrain.py  gate.py
  configs/
    sweep.yaml             the hyperparameter grid, explicit and readable
    thresholds.yaml        gate thresholds, a comment per line on when and why it was set
    drift.yaml             future-batch rule, PSI thresholds, promotion margin
  tests/                   pytest: unit tests per step, one integration test on a 500-row sample
  reports/                 committed outputs: sweep table, drift tables, error analysis, plots
  docs/
    brief.md               this file
    status.md              what is done and what is not, updated every session
    decisions.md           short dated decision records
    preflight.md           the ten-item checklist with evidence links, filled in Session 3
    reference/             field-guide-rules.md and the two source HTML files
  .github/workflows/ci.yml
  requirements.txt  pyproject.toml  README.md  LICENSE  .env.example  .gitignore
```

## Commands (python -m mlops_loop ...)
skeleton    steps 1 to 8 once, one model, everything logged
train       tracked sweep over configs/sweep.yaml
eval        gate against configs/thresholds.yaml, exit 1 on failure
drift       monitor the future batch, retrain and challenge if drifted
serve       uvicorn on :8000
ui          mlflow ui on :5000
reproduce   skeleton, train, eval, drift, reports, from a clean clone

## Phases
Session 1: scaffold, decomposition committed, skeleton end to end, repo public.
Session 2: tracked sweep, registry discipline, eval gate, CI green and seen to fail once on purpose.
Session 3: drift, retrain trigger, champion versus challenger, error analysis, README with real numbers, preflight filled.
Session 4: Track B in order-processing-workflow.

## Definition of done (whole project)
- `python -m mlops_loop reproduce` works from a clean clone with only Python installed, in under 10 minutes, with no network beyond the dataset URL and PyPI.
- MLflow shows at least 8 tracked runs with differing params, one registered model with at least 2 versions, a champion alias, and at least 2 logged promotion decisions, whichever way they went.
- CI is green, and the gate has been seen to fail at least once on purpose. The failed run's link is in docs/status.md.
- README shows the sweep table and the drift table from logged runs, with run ids.
- reports/error_analysis.md attributes every sampled failure to a component and names the biggest cell.
- docs/preflight.md has every item either ticked with an evidence link or left unticked with a reason.
- No secret in the repo history. .env gitignored from the first commit.

## Non-goals
Kubernetes, Kubeflow, a feature store product, a hosted tracking server, deep learning, hyperparameter optimisation libraries, dashboards beyond the MLflow UI. If a later session wants one of these, it is a new decision record, not scope creep.
