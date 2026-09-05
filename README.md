# mlops-loop

[![ci](https://github.com/GabrieleBosi/mlops-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrieleBosi/mlops-loop/actions/workflows/ci.yml)

One tabular ML system taken through the whole lifecycle on MLflow: tracked experiments, a model
registry with champion and challenger promotion, drift-triggered retraining, a serving endpoint
that loads from the registry, and a CI gate that fails closed. The model predicts telco customer
churn on the IBM Telco Customer Churn file, 7,043 customers and 21 columns.

The point is not the model. The point is that every number in this README comes out of a logged
run and cites its run id, that the pipeline stops and says why when the data is not what it
expects, and that the whole loop rebuilds from a clean clone in under two minutes.

Every step is deterministic Python. No LLM, no agent framework. The decomposition table in
[docs/brief.md](docs/brief.md) assigns an executor to each of the eleven steps and the answer is
"code" in every row, because code is free, exact and testable. A second track applies the same
tracking discipline to a GenAI evaluation suite: 39 cases including safety and multilingual
ones, per-category thresholds gated in CI, MLflow-traced, with a judge calibrated against
labels and the disagreements published. It lives in
[order-processing-workflow](https://github.com/GabrieleBosi/order-processing-workflow#evaluation).

## The pipeline

| # | Step | Input to output | Where |
|---|------|-----------------|-------|
| 1 | Ingest | CSV URL to `data/raw.parquet`, sha256 of the raw bytes logged | `ingest.py` |
| 2 | Validate | raw frame to a validated frame, or a stop with the violation listed | `validate.py` |
| 3 | Split | validated frame to train, val, holdout and two future batches, keyed by customerID | `split.py` |
| 4 | Features | frame to matrix, feature-code hash and column list logged | `features.py` |
| 5 | Train | 14 explicit configs, two families, one child run each | `train.py` |
| 6 | Evaluate | ROC-AUC, PR-AUC, recall at precision 0.5, Brier, calibration plot | `evaluate.py` |
| 7 | Register | winner registered, `champion` alias moved, outgoing one kept as `previous` | `register.py` |
| 8 | Serve | JSON to prediction, model version and run id | `serve.py` |
| 9 | Monitor | future batch to PSI per feature and on the prediction distribution | `monitor.py` |
| 10 | Retrain trigger | PSI over threshold to challenger, compared on the fixed holdout, promoted or rejected | `retrain.py` |
| 11 | Gate | champion against `configs/thresholds.yaml`, non-zero exit | `gate.py` |

## Quickstart

```
py -3.12 -m venv .venv
.venv\Scripts\activate            # Linux and macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m mlops_loop reproduce    # the whole loop, about 65 seconds
```

Or one step at a time:

```
python -m mlops_loop skeleton     # steps 1 to 8 once, one model, everything logged
python -m mlops_loop train        # the sweep: 14 configs, select, register, promote
python -m mlops_loop eval         # the gate: exits 1 if the champion is below threshold
python -m mlops_loop drift        # PSI per batch, challenge the champion where it drifted
python -m mlops_loop reports      # error analysis on the champion
python -m mlops_loop ui           # MLflow UI on :5000, read the runs
python -m mlops_loop serve        # FastAPI on :8000
pytest -q                         # 109 tests, offline, on a committed 500-row fixture
```

The MLflow backend is `sqlite:///mlflow.db` with artifacts under `./mlruns`, both gitignored, so
a clean clone starts from nothing and rebuilds its own history. The only network calls are the
dataset URL and PyPI.

## Reproducing this README

Everything below came from one `python -m mlops_loop reproduce` at commit `0b1cd3a`, 63.5
seconds on a Windows laptop:

| stage | seconds | result |
|-------|--------:|--------|
| skeleton | 22.0 | run `74e4cde1680b4a78ad06d270e7c53fd6`, registered version 1 |
| train | 19.7 | parent `a15a64efaaf94f8dbbe980cec4ea13f3`, 14 configs, version 2 |
| eval | 2.3 | run `07797376000849eab75e3137d91fe62e`, 4 of 4 passed |
| drift | 14.8 | 2 batches, 2 decisions, 1 promotion |
| reports | 2.6 | run `ac4850f895b5402e898cc31e97588504` |
| eval (final champion) | 2.1 | run `ea66c83c2d454c329c46145c48340c4a`, version 3, passed |

From a genuinely clean clone with an empty registry, on the same commit: 101 seconds to clone,
build a venv and install, then 84 seconds for `reproduce`, and every metric below came back
identical to four decimal places.

## The sweep

Selection is PR-AUC on the validation split. The holdout is looked at exactly once, by the
winner, after selection is over. Full table with every child run id:
[reports/sweep_comparison.md](reports/sweep_comparison.md).

| rank | config | family | val PR-AUC | val ROC-AUC | val Brier |
|-----:|--------|--------|-----------:|------------:|----------:|
| 1 | `lr-c0.05-balanced` (won) | logistic_regression | 0.5851 | 0.8094 | 0.1868 |
| 2 | `lr-c0.05` | logistic_regression | 0.5731 | 0.7976 | 0.1604 |
| 3 | `lr-c0.5` | logistic_regression | 0.5667 | 0.8088 | 0.1595 |
| 5 | `lr-c1.0` (the incumbent) | logistic_regression | 0.5423 | 0.8004 | 0.1655 |
| 8 | `hgb-lr0.10-leaf15-l2` (best tree) | hist_gradient_boosting | 0.4623 | 0.7226 | 0.1998 |
| 14 | `lr-c5.0-balanced` | logistic_regression | 0.3999 | 0.6932 | 0.2020 |

Every linear model but one beat every tree. The training set contains no fibre-optic customer
while 43.6 % of the holdout is one, and a tree cannot split on a one-hot column that is constant
zero, so it scores those customers as ordinary. The linear model extrapolates its coefficient and
at least moves them in the right direction.

## Drift and the two promotion decisions

`configs/drift.yaml` defines two future batches as real feature slices, never injected noise.
Both are carved out at split time and no model trains on them until a trigger says so. PSI is
computed in numpy against the training split, ten quantile bins for numeric columns and one bin
per value otherwise, with the conventional reading: below 0.10 none, 0.10 to 0.25 moderate, above
0.25 major.

| batch | rows | churn | monitor run | PSI breaches | champion PR-AUC on the batch |
|-------|-----:|------:|-------------|-------------:|-----------------------------:|
| `fibre-monthly` | 1,451 | 0.5410 | `c934991a348d4126a48cf7301d99296d` | 16 of 20 | 0.7139 |
| `fibre-committed` | 663 | 0.1207 | `cfa00a696f894394b432a60020c9288b` | 16 of 20 | 0.1956 |

Loudest signals on both: `InternetService` 26.97, `MonthlyCharges` 10.67 and 11.36, `Contract`
7.46 and 6.04. The prediction distribution moved 4.3735 on the first batch and 0.9595 on the
second, against a threshold of 0.10.

The two batches are the same drift in the inputs and the opposite drift in the labels: one churns
at 3.5 times the training rate, the other slightly below it. A monitor that only watches inputs
cannot tell them apart, which is why the trigger only earns the right to train a challenger and
the holdout decides what happens next.

**Decision 1, `fibre-monthly`, run `853f20f3e0d64adab9ea9158ca970bb6`: promoted.** Challenger run
`5873c3dcee694a6a9b32316f882662b9`, the champion's own configuration refitted on 4,125 rows.

| metric | champion v2 | challenger | delta |
|--------|------------:|-----------:|------:|
| PR-AUC | 0.5944 | 0.6515 | **+0.0571** |
| ROC-AUC | 0.8014 | 0.8258 | +0.0244 |
| recall at precision 0.5 | 0.6893 | 0.7750 | +0.0857 |
| Brier | 0.1962 | 0.1643 | −0.0319 |

+0.0571 is more than the 0.01 margin, so the challenger became version 3 and took `champion`;
version 2 kept its version and took `previous`.

**Decision 2, `fibre-committed`, run `7d1ee8ce585d4e05ac66fc3bfcd387b1`: rejected.** Challenger
run `1039e97ee6fa43a087271e30d4cfa583`, refitted on 3,337 rows.

| metric | champion v3 | challenger | delta |
|--------|------------:|-----------:|------:|
| PR-AUC | 0.6515 | 0.6500 | **−0.0015** |
| ROC-AUC | 0.8258 | 0.8334 | +0.0076 |
| recall at precision 0.5 | 0.7750 | 0.8000 | +0.0250 |
| Brier | 0.1643 | 0.2272 | +0.0628 |

This is the more interesting one. Two of four metrics improved and the challenger would look
like a win to anyone reading ROC-AUC. The rule is PR-AUC past a margin, it came back at −0.0015,
and the calibration got much worse, so the champion kept the alias. Neither outcome was
engineered: the rule was fixed in `configs/drift.yaml` before either batch ran.

Per-batch detail: [reports/drift_fibre-monthly.md](reports/drift_fibre-monthly.md),
[reports/drift_fibre-committed.md](reports/drift_fibre-committed.md).

## The gate

`python -m mlops_loop eval` loads `models:/churn@champion`, scores the fixed holdout and exits 1
on any failure. Thresholds are in [configs/thresholds.yaml](configs/thresholds.yaml), one comment
per line saying which run it came from. Final gate run `ea66c83c2d454c329c46145c48340c4a` on
version 3:

| metric | value | threshold | margin | result |
|--------|------:|-----------|-------:|--------|
| `holdout_roc_auc` | 0.8258 | >= 0.7699 | +0.0559 | pass |
| `holdout_pr_auc` | 0.6515 | >= 0.5164 | +0.1351 | pass |
| `holdout_recall_at_precision_50` | 0.7750 | >= 0.5964 | +0.1786 | pass |
| `holdout_brier` | 0.1643 | <= 0.1981 | +0.0338 | pass |

The gate has been seen to fail on purpose, on a branch since deleted:
[run 33948528226](https://github.com/GabrieleBosi/mlops-loop/actions/runs/33948528226), exit
code 1, with `pytest` still green so the signal pointed at the gate rather than the code.

## Error analysis

Every holdout row the champion gets wrong at threshold 0.5, attributed by code to the first
component that could have prevented it. Analysis run `ac4850f895b5402e898cc31e97588504`, champion
version 3, 254 of 1,057 rows wrong (24.0 %): 72 false negatives, 182 false positives.

| component | sampled (20) | all (254) | what it means |
|-----------|-------------:|----------:|---------------|
| split | 0 | 1 | outside the region the champion's training rows cover |
| features | 2 | 40 | the 25 nearest training neighbours churn at the base rate, so there is no signal |
| **model** | **17** | **188** | in distribution, informative neighbourhood, ranked wrongly anyway |
| threshold | 1 | 25 | ranked correctly, only the 0.5 cut is wrong |

The biggest cell is `model` and the failures there are not near misses. 140 of the 188 are false
positives at a median predicted probability of 0.739: the model says these customers are leaving
and they stay. The champion flags 36.9 % of the holdout as churners against an actual 26.5 %,
which is what `class_weight='balanced'` does, and PR-AUC, the metric that selected it, cannot see
the cost because it only reads the ranking.

Full table with all twenty customerIDs and the rule that charged each one:
[reports/error_analysis.md](reports/error_analysis.md).

## What I would do next

Grounded in that tally, cheapest first.

**Put calibration into the selection rule.** The biggest cell is `model` and its shape is
over-prediction, so the fix is to stop selecting on ranking alone: add Brier as a constraint
alongside the PR-AUC objective in `configs/sweep.yaml` and re-run `train`. The alternative is
already measured rather than hypothetical, because the sweep contains the unweighted twin of the
current champion: `lr-c0.05` trades 0.012 of val PR-AUC for 0.026 of val Brier. Two lines of
config, no new code, 26 seconds of compute, and falsifiable: if `model` is still the biggest cell
afterwards, the diagnosis was wrong and capacity or features come next, in that order.

**Add a component-level calibration test.** `docs/preflight.md` leaves item 9 unticked for a
reason: 109 tests cover every step except the one the error analysis blames. Nothing currently
fails when calibration degrades; the gate's Brier line is a system-level check standing in for a
component-level one, and in Session 2 it passed the champion by 0.0019.

**Fold the promoted challenger's data back into the default training set.** The `split` cell is
almost empty now, 1 of 254, precisely because drift retraining put fibre-monthly customers into
the champion's training data. That happened as a side effect of a drift trigger rather than by
design, and a model that serves the whole population should be fitted on it from the start.

**Then promotion on merit in the sweep.** `train` still gives `champion` to the sweep winner
without a head-to-head test against the incumbent, which is the one place in this repo where a
worse model can take the alias with only the gate to stop it. `drift` already does the
comparison properly and `promotion.json` records both sides; the rule just needs moving.

## Serving

`python -m mlops_loop serve` loads `models:/churn@champion` from the registry at startup.

```
GET  /health   -> {"status": "ok", "model_name", "model_version", "run_id"}
POST /predict  -> {"prediction", "probability", "model_name", "model_version", "run_id"}
```

A `/predict` body goes through the same pandera schema the training data went through, minus the
target. An unknown category returns 422 with the violation named, it does not return a guess.

## Design notes worth knowing

The split order is holdout, then val, then the future batches, then train. The holdout is drawn
first, stratified on churn over the whole population, and is scored but never trained on.
`split.assert_holdout_unseen` compares customerIDs before any fit and raises on overlap, so that
rule is enforced rather than assumed.

Val is drawn next, from the same population, because a validation split has to look like the data
the model will serve or it cannot rank candidates by anything that matters. Session 1 drew it
from the reference pool instead, which contains no fibre-optic customer, and Session 2's sweep
selected a model whose holdout ROC-AUC was 0.63. Val now matches the holdout on churn rate,
0.2654 against 0.2649, and on fibre share, 0.435 against 0.436.

The holdout is handed out by `split.HoldoutBudget`, which allows a fixed number of looks, records
why each one was taken, and raises on the next. The sweep gets one, for the winner, after
selection is over. A promotion decision gets two, one per model. Scoring every candidate on the
holdout and keeping the best would report the maximum of 14 samples rather than an estimate.

One-hot categories are pinned to the sets the schema validates against, not learned from the fit
data, so a model trained without a single fibre-optic customer still produces an
`InternetService_Fiber optic` column and can be compared with one that was.

Decisions and their reasons are in [docs/decisions.md](docs/decisions.md). What exists and what
does not: [docs/status.md](docs/status.md). The pre-flight checklist with evidence links:
[docs/preflight.md](docs/preflight.md).

## Status

Built: all eleven steps of the decomposition table, the CLI, the FastAPI endpoint, the registry
with `champion` and `previous` aliases, the tracked sweep, drift monitoring on two batches, the
retrain trigger with champion-versus-challenger promotion, the eval gate, 109 tests, and CI that
rebuilds the registry from nothing and runs the whole loop on every push and weekly.

Not built: promotion on merit inside the sweep, calibration in the selection rule, and a
component-level calibration test. All three are argued for above. Track B, the GenAI
evaluation suite, is built and linked at the top of this file.

## Licence

MIT. See [LICENSE](LICENSE).
