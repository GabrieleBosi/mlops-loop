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
| 5 | Train | 14 explicit configs, two families, one child run each | `train.py` |
| 6 | Evaluate | ROC-AUC, PR-AUC, recall at precision 0.5, Brier, calibration plot | `evaluate.py` |
| 7 | Register | winner registered, `champion` alias moved, outgoing one kept as `previous` | `register.py` |
| 8 | Serve | JSON to prediction, model version and run id | `serve.py` |

| 11 | Gate | champion vs `configs/thresholds.yaml` on the holdout, non-zero exit | `gate.py` |

Steps 9 and 10, drift monitoring and the retrain trigger, are not built yet. See Status.

## Quickstart

```
py -3.12 -m venv .venv
.venv\Scripts\activate            # Linux and macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m mlops_loop skeleton     # steps 1 to 8 once, one model, everything logged
python -m mlops_loop train        # the sweep: 14 configs, select, register, promote
python -m mlops_loop eval         # the gate: exits 1 if the champion is below threshold
python -m mlops_loop ui           # MLflow UI on :5000, read the runs
python -m mlops_loop serve        # FastAPI on :8000
pytest -q                         # offline, no network, on the committed 500-row fixture
```

`skeleton` fetches the dataset on every run and writes it to `data/`, which is gitignored, so
the sha256 on the run is always the digest of the bytes that run actually used.
The MLflow backend is `sqlite:///mlflow.db` with artifacts under `./mlruns`, both gitignored, so a
clean clone starts from nothing and rebuilds its own history.

Calling a command that is not built yet prints which session adds it and exits 2, rather than
quietly doing nothing.

## What one run logs

Params: `dataset_source`, `dataset_sha256`, `feature_code_hash`, `n_encoded_features`, `split_seed`,
`holdout_fraction`, `val_fraction`, `future_rule`, `model_class`, `model_family` and one `model_*`
per hyperparameter.

Metrics: `raw_rows`, `total_charges_blank_count`, `rows_*` and `churn_rate_*` per split,
`val_*`, `holdout_*` and `future_*` for the four evaluation metrics, and `seconds_*` per step.

Tags: `git_commit`, `git_dirty`, `phase`, `model_name`, `model_version`, `model_alias`, and
`step_<name>` for each step, so a failure names the step that stopped.

Artifacts: `calibration_val.png` and `calibration_holdout.png`, `split_ids.json`,
`feature_columns.json`, `serve_check.json`, `promotion.json`, `gate.json`, the configs, the
registered model, and the dataset input.

A sweep is one parent run holding the data steps and the outcome, with one child run per config
holding that config's params, its validation metrics and its calibration curve. Only the winner's
model is stored, logged into the child run that trained it, so the registry version points at the
run that produced it.

## The sweep and the gate

Sweep parent run `4afa568e01354c78acc720694a94a99c`, commit `be86fd5`, 14 configs in 25.7 seconds.
Selection is validation PR-AUC. The full table with every child run id is in
[reports/sweep_comparison.md](reports/sweep_comparison.md).

| rank | config | family | val PR-AUC | val ROC-AUC | val Brier |
|-----:|--------|--------|-----------:|------------:|----------:|
| 1 | `lr-c0.05-balanced` (champion) | logistic_regression | 0.5851 | 0.8094 | 0.1868 |
| 2 | `lr-c0.05` | logistic_regression | 0.5731 | 0.7976 | 0.1604 |
| 3 | `lr-c0.5` | logistic_regression | 0.5667 | 0.8088 | 0.1595 |
| 5 | `lr-c1.0` (the incumbent) | logistic_regression | 0.5423 | 0.8004 | 0.1655 |
| 8 | `hgb-lr0.10-leaf15-l2` (best tree) | hist_gradient_boosting | 0.4623 | 0.7226 | 0.1998 |
| 14 | `lr-c5.0-balanced` | logistic_regression | 0.3999 | 0.6932 | 0.2020 |

Every linear model but one beat every tree. The training set contains no fibre-optic customer
while 43.6 % of the holdout is one, and a tree cannot split on a one-hot column that is constant
zero, so it scores those customers as ordinary. The linear model extrapolates its coefficient and
at least moves them in the right direction.

The winner became `churn` version 2 and took the `champion` alias; version 1 kept its version and
took `previous`. The holdout was looked at exactly once, by the winner, after selection was over.

`python -m mlops_loop eval` on that champion, gate run `de2de7f753cc45ea87d42752d2b32d9c`:

| metric | value | threshold | margin | result |
|--------|------:|-----------|-------:|--------|
| `holdout_roc_auc` | 0.8014 | >= 0.7699 | +0.0315 | pass |
| `holdout_pr_auc` | 0.5944 | >= 0.5164 | +0.0780 | pass |
| `holdout_recall_at_precision_50` | 0.6893 | >= 0.5964 | +0.0929 | pass |
| `holdout_brier` | 0.1962 | <= 0.1981 | +0.0019 | pass |

Read the last line. Selecting on PR-AUC rewards ranking and is indifferent to whether the
probabilities mean anything, so it picked the reweighted model, which ranks slightly better and is
calibrated noticeably worse than the `lr-c0.05` directly below it. The gate caught it by two
thousandths. That is an argument for putting calibration into the selection rule, not just the
gate.

The thresholds are in [configs/thresholds.yaml](configs/thresholds.yaml), one comment per line
saying which run it came from. They are a regression gate against the incumbent, and the tolerance
rule was committed before the sweep existed.

To reproduce: `python -m mlops_loop train` then `python -m mlops_loop eval` at commit `be86fd5`.
Everything is seeded, so the rows and the numbers land the same way.

## Serving

`python -m mlops_loop serve` loads `models:/churn@champion` from the registry at startup.

```
GET  /health   -> {"status": "ok", "model_name", "model_version", "run_id"}
POST /predict  -> {"prediction", "probability", "model_name", "model_version", "run_id"}
```

A `/predict` body goes through the same pandera schema the training data went through, minus the
target. An unknown category returns 422 with the violation named, it does not return a guess.

## Design notes worth knowing

The split order is holdout, then val, then the future batch, then train. The holdout is drawn
first, stratified on churn over the whole population, and is scored but never trained on.
`train.fit_model` asserts the holdout customerIDs are absent from the data it is about to fit, so
that rule is enforced rather than assumed.

Val is drawn next, from the same population, because a validation split has to look like the data
the model will serve or it cannot rank candidates by anything that matters. Session 1 drew it from
the reference pool instead, which contains no fibre-optic customer, and Session 2's sweep selected
a model whose holdout ROC-AUC was 0.63. Val now matches the holdout on churn rate, 0.2654 against
0.2649, and on fibre share, 0.435 against 0.436.

Selection touches val only. The holdout is handed out by `split.HoldoutBudget`, which allows one
look per sweep, records why, and raises on the second. Scoring every candidate on the holdout and
keeping the best would report the maximum of 14 samples rather than an estimate.

The future batch is a real slice, not injected noise: 2,114 fibre-optic customers carved out at
split time, who churn at 40.9 % against 15.2 % for the training set. The champion never sees one
during training. Session 3 uses it for drift detection and the challenger comparison.

One-hot categories are pinned to the sets the schema validates against, not learned from the fit
data. That is what makes the comparison in Session 3 possible: a model trained without a single
fibre-optic customer still produces an `InternetService_Fiber optic` column, so champion and
challenger have the same matrix shape and can be scored on the same holdout.

Decisions and their reasons are in [docs/decisions.md](docs/decisions.md).

## Status

Built (Sessions 1 and 2): steps 1 to 8 and step 11. The CLI, the FastAPI endpoint, the tracked
sweep, the registry with `champion` and `previous` aliases, the eval gate, the test suite, and CI
that rebuilds the registry and runs the gate on every push and weekly.

Not built yet:

- Promotion on merit: the sweep winner takes `champion` without a head-to-head test against the
  current champion on the fixed holdout (Session 3)
- `drift`: PSI per feature and on the prediction distribution, over the future batch (Session 3)
- The retrain trigger and the champion-versus-challenger comparison (Session 3)
- `reproduce`: the whole loop from a clean clone (Session 3)
- `reports/error_analysis.md` (Session 3)
- `docs/preflight.md` (Session 3)
- Track B, the GenAI evaluation suite in order-processing-workflow (Session 4)

Current state and the run ids behind it: [docs/status.md](docs/status.md).

## Licence

MIT. See [LICENSE](LICENSE).
