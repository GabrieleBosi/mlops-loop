# mlops-loop

[![ci](https://github.com/GabrieleBosi/mlops-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrieleBosi/mlops-loop/actions/workflows/ci.yml)

One tabular ML system taken through the whole lifecycle on MLflow. The model predicts telco customer
churn on the IBM Telco Customer Churn file: 7,043 customers, 21 columns, a binary target. The point
is not the model. The point is that every claim in this README comes out of a logged run, and that
the pipeline stops and says why when the data is not what it expects.

Every step is deterministic Python. No LLM, no agent framework. The decomposition table in
[docs/brief.md](docs/brief.md) assigns an executor to each of the eleven steps and the answer is
"code" in every row, because code is free, exact and testable. A second track applies the same
tracking discipline to a GenAI evaluation suite; it lives in the order-processing-workflow repo and
will be linked here in Session 4.

## The pipeline

| # | Step | Input to output | Where |
|---|------|-----------------|-------|
| 1 | Ingest | CSV URL to `data/raw.parquet`, sha256 of the raw bytes logged | `ingest.py` |
| 2 | Validate | raw frame to a validated frame, or a stop with the violation listed | `validate.py` |
| 3 | Split | validated frame to train, val, holdout and a future batch, keyed by customerID | `split.py` |
| 4 | Features | frame to matrix, feature-code hash and column list logged | `features.py` |
| 5 | Train | one LogisticRegression pipeline (the sweep lands in Session 2) | `train.py` |
| 6 | Evaluate | ROC-AUC, PR-AUC, recall at precision 0.5, Brier, calibration plot | `evaluate.py` |
| 7 | Register | model version in the registry, `champion` alias moved | `register.py` |
| 8 | Serve | JSON to prediction, model version and run id | `serve.py` |

Steps 9 to 11, monitoring, the retrain trigger and the eval gate, are not built yet. See Status.

## Quickstart

```
py -3.12 -m venv .venv
.venv\Scripts\activate            # Linux and macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m mlops_loop skeleton     # steps 1 to 8 once, everything logged
python -m mlops_loop ui           # MLflow UI on :5000, read the run
python -m mlops_loop serve        # FastAPI on :8000
pytest -q                         # offline, no network, on the committed 500-row fixture
```

`skeleton` downloads the dataset on the first run and writes it to `data/`, which is gitignored.
The MLflow backend is `sqlite:///mlflow.db` with artifacts under `./mlruns`, both gitignored, so a
clean clone starts from nothing and rebuilds its own history.

Calling a command that is not built yet prints which session adds it and exits 2, rather than
quietly doing nothing.

## What one run logs

Params: `dataset_source`, `dataset_sha256`, `feature_code_hash`, `n_encoded_features`, `split_seed`,
`holdout_fraction`, `val_fraction`, `future_rule`, `model_class` and one `model_*` per
hyperparameter.

Metrics: `raw_rows`, `total_charges_blank_count`, `rows_*` and `churn_rate_*` per split,
`val_*` and `holdout_*` for the four evaluation metrics, and `seconds_*` per step.

Tags: `git_commit`, `git_dirty`, `phase`, `model_name`, `model_version`, `model_alias`, and
`step_<name>` for each of the eight steps, so a failure names the step that stopped.

Artifacts: `calibration_holdout.png`, `split_ids.json`, `feature_columns.json`, `serve_check.json`,
the two configs, the registered model, and the dataset input.

## The skeleton run

Run `ae4f0cd88d4b4d439bbcadb545bd01b6`, experiment `churn`, commit `605ab74`, `git_dirty=false`.
Dataset sha256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` over 7,043 rows,
of which 11 had a blank `TotalCharges` and were set to 0.0 after the tenure-0 check. Feature-code
hash `215d7d44d683a290ba5e4a3561fd2f5aa75b4b960203a84643ad6542a4205b49`, 45 encoded columns.
One LogisticRegression, `C=1.0`, `solver=lbfgs`, `max_iter=1000`, registered as `churn` version 1
with the `champion` alias.

| split | rows | churn rate | ROC-AUC | PR-AUC | recall at precision 0.5 | Brier |
|-------|-----:|-----------:|--------:|-------:|------------------------:|------:|
| train | 2,680 | 0.1466 | fitted on | | | |
| val | 671 | 0.1475 | 0.8448 | 0.5072 | 0.5657 | 0.0949 |
| holdout | 1,057 | 0.2649 | 0.8186 | 0.5971 | 0.7536 | 0.1493 |
| future | 2,635 | 0.4163 | 0.7721 | 0.6618 | 0.9572 | 0.2067 |

Read the val and holdout columns together. Val comes from the same reference pool the model was fit
on, so its churn rate is 0.1475 and its Brier is low because most predictions are confidently near
zero. The holdout is drawn from the whole population, churns at 0.2649, and is the number to quote.

The future batch is the reason the rest of this repo exists. The model has never seen a fibre-optic
customer, and on that batch its Brier score is 0.2067 against 0.1493 on the holdout: the ranking
still works, ROC-AUC 0.7721, but the probabilities are badly calibrated. Session 3 has to catch that
without being told the answer.

To reproduce these numbers: `python -m mlops_loop skeleton` at commit `605ab74`, then
`python -m mlops_loop ui` and open the run. The split is seeded, so the rows land the same way.

## Serving

`python -m mlops_loop serve` loads `models:/churn@champion` from the registry at startup.

```
GET  /health   -> {"status": "ok", "model_name", "model_version", "run_id"}
POST /predict  -> {"prediction", "probability", "model_name", "model_version", "run_id"}
```

A `/predict` body goes through the same pandera schema the training data went through, minus the
target. An unknown category returns 422 with the violation named, it does not return a guess.

## Design notes worth knowing

The holdout is drawn first, stratified on churn over the whole population, and is scored but never
trained on. `train.fit_model` asserts the holdout customerIDs are absent from the data it is about
to fit, so that rule is enforced rather than assumed.

The future batch is a real slice, not injected noise: fibre-optic customers, carved out at split
time, who churn at 41.6 % against 14.7 % for the reference set. The champion never sees one during
training. Session 3 uses it for drift detection and the challenger comparison.

One-hot categories are pinned to the sets the schema validates against, not learned from the fit
data. That is what makes the comparison in Session 3 possible: a model trained without a single
fibre-optic customer still produces an `InternetService_Fiber optic` column, so champion and
challenger have the same matrix shape and can be scored on the same holdout.

Decisions and their reasons are in [docs/decisions.md](docs/decisions.md).

## Status

Built (Session 1): steps 1 to 8, the CLI, the FastAPI endpoint, the registry with a `champion`
alias, the test suite and CI.

Not built yet:

- `train`: the tracked sweep over `configs/sweep.yaml` (Session 2)
- `eval`: the gate against `configs/thresholds.yaml`, exiting non-zero below threshold (Session 2)
- Promotion on merit: Session 1 promotes its single model unconditionally (Session 2)
- `drift`: PSI on the future batch, the retrain trigger, champion versus challenger (Session 3)
- `reproduce`: the whole loop from a clean clone (Session 3)
- `reports/`: the sweep table, drift tables and error analysis (Session 3)
- `docs/preflight.md` (Session 3)
- Track B, the GenAI evaluation suite in order-processing-workflow (Session 4)

Current state and the run ids behind it: [docs/status.md](docs/status.md).

## Licence

MIT. See [LICENSE](LICENSE).
