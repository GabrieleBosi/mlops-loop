# Decisions

Short, dated. One paragraph each: what was decided, what it replaces, why.

## 2026-09-04 Tracking tool: MLflow
MLflow over Weights & Biases. Self-hosted with sqlite, so the repo reproduces offline; one tool covers tracking, registry, GenAI evaluation and tracing. W&B would give a shareable hosted dashboard; not worth a second tool at this scope.

## 2026-09-04 Dataset: IBM Telco Customer Churn
Over UCI student dropout. Same size class, but churn has a natural drift story when sliced by contract or internet-service type.

## 2026-09-04 Autonomy level: 2, no LLM in Track A
Every step in the decomposition table is deterministic code. An LLM would add cost and non-determinism without doing any step better. Track B is where an LLM is the only executor that can do the step.

## 2026-09-04 No make
Windows development machine without make. The CLI is `python -m mlops_loop <command>`; CI calls the same commands so there is one path.

## 2026-09-05 Blank TotalCharges: 0.0 when tenure is 0, stop otherwise
All 11 blank `TotalCharges` in the source file are customers with `tenure == 0`, and they are the
only tenure-0 rows. A customer who has been billed for zero months has been billed 0.00, so the
blank is set to 0.0 and the count is logged as the metric `total_charges_blank_count`. A blank on a
row with `tenure != 0` means the source changed, and validate stops with the offending customerIDs.
This replaces the common `dropna()` or `fillna(median)`, both of which hide the fact silently.

## 2026-09-05 Feature encoding: fixed category lists shared with the schema
The 15 categorical columns are one-hot encoded with `categories=` pinned to the sets captured from
the source file, `handle_unknown="error"`, no dropped level. The 4 numeric columns
(`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) are standardised. `customerID` is an
identifier, not a feature, and is excluded; so are `Churn` and `churn`. 45 encoded columns.
The category sets live in one dict, `validate.CATEGORIES`, which the pandera schema checks against
and `features.py` builds the encoder from, so the contract and the encoder cannot drift apart.
Pinning the categories rather than learning them from the fit data is what makes Session 3 possible:
a model trained on the reference split, which contains no fibre-optic customer, still produces an
`InternetService_Fiber optic` column of zeros, so champion and challenger have the same matrix shape
and can be scored on the same holdout.

## 2026-09-05 Split: holdout 15 % of the whole population, then the future batch, then train/val
Order matters and is fixed. The holdout is drawn first, stratified on churn over all 7,043 rows, so
it keeps the population mix (26.5 % churn, 43.6 % fibre) and stays a fair scoreboard for any model
trained later, including a challenger trained on the future batch. The future batch is carved from
what remains. The rest is the reference set, split 80/20 into train and val, stratified. Seed 42,
keyed by `customerID`, and the four id sets are asserted pairwise disjoint and covering. Sizes on
the full file: train 2,680, val 671, holdout 1,057, future 2,635. `train.fit_model` asserts the
holdout ids are absent from the data it is about to fit on, so "scored, never trained on" is
enforced rather than assumed.

## 2026-09-05 Future batch for Session 3: InternetService == "Fiber optic"
Chosen over injected noise and over the top tenure quartile. The shift is real and explainable:
fibre customers churn at 41.6 % against 14.7 % for the reference set, and the champion never sees
one during training. Measured on run `ae4f0cd88d4b4d439bbcadb545bd01b6`, the reference champion scores holdout
Brier 0.1493 and future-batch Brier 0.2067, which is the degradation Session 3's PSI monitor and
challenger comparison are meant to catch. Fallback if this turns out too easy: top tenure quartile
(tenure > 55, 1,755 rows, 7.8 % churn).

## 2026-09-05 Skeleton model: one LogisticRegression, promoted unconditionally
`C=1.0, solver=lbfgs, max_iter=1000, class_weight=None, random_state=42`. Session 1 has exactly one
model, so `register()` moves the `champion` alias without comparing anything. Session 2 replaces the
`promote` flag with a comparison against the current champion on the fixed holdout. Recorded here so
the unconditional promotion is a known gap, not an oversight.

## 2026-09-05 Metric definitions
ROC-AUC; PR-AUC is `average_precision_score`, not the trapezoid under an interpolated PR curve;
`recall_at_precision_50` is the highest recall over all thresholds whose precision is at least 0.5,
read off `precision_recall_curve`, and 0.0 when no threshold reaches that precision; Brier is
`brier_score_loss` on the churn probability. Any non-finite metric stops the run rather than being
logged. The calibration plot uses 10 quantile bins, so each point rests on the same number of
customers.

## 2026-09-05 New dependencies: matplotlib, httpx
`matplotlib==3.11.1` for the calibration plot, used with the Agg backend so it never needs a
display. It replaces nothing; the alternative was logging calibration as a table only, which is
harder to read at a glance. `httpx==0.28.1` is required by `fastapi.testclient.TestClient` and is a
test-only dependency; it replaces spinning up a real uvicorn process in the test suite.

## 2026-09-05 pandas pinned to 2.3.3
The 2.3 line, not 3.x. mlflow 3.16, pandera 0.33 and pyarrow 25 are all exercised against pandas 2
in their own CI, and a pandas 3 upgrade is a change worth making on its own evidence rather than
folded into the first commit. Revisit when Session 2 touches requirements.

## 2026-09-05 Committed 500-row test fixture
`tests/fixtures/telco_sample_500.csv` is the 11 blank-TotalCharges rows plus 489 rows sampled with
seed 42 from the source file, sorted by customerID. Generated once with:

    raw = pd.read_csv("data/raw.csv", dtype=str, keep_default_na=False)
    blank = raw[raw.TotalCharges.str.strip() == ""]
    rest = raw.drop(blank.index).sample(n=489, random_state=42)
    pd.concat([blank, rest]).sort_values("customerID", kind="mergesort").to_csv(
        "tests/fixtures/telco_sample_500.csv", index=False, lineterminator="\n")

Every category of every column appears at least once, and both classes appear in every split, so the
integration test exercises the real encoder. It is committed so pytest and CI need no network and
give the same answer on every machine.

## 2026-09-05 Digests: dataset sha256 over raw bytes, feature-code hash over features.py
`dataset_sha256` is the sha256 of the downloaded CSV bytes, taken before parsing, so it changes the
moment the source file changes. `feature_code_hash` is the sha256 of `mlops_loop/features.py`, so
any change to the encoding shows up as a different hash on the run. Both are run params, alongside
the `git_commit` tag, which is what makes a run reproducible from the run alone.

## 2026-09-05 Validation split drawn from the population, not the reference pool
Session 1 drew val from the reference rows, after the fibre-optic future batch had been
removed. That was wrong and Session 2's sweep exposed it. The holdout is 43.6 % fibre and val
was 0 % fibre, so val scored every candidate on a sub-population the champion does not serve.
Selecting on it picked `hgb-lr0.05-leaf15-balanced`, val ROC-AUC 0.8342, whose holdout ROC-AUC
was 0.6281 against the linear model's 0.80.

Error analysis before the fix, as the field guide requires. Attribution: step 3, Split, is the
first failing component. On holdout slices every candidate is comparable, non-fibre ROC-AUC
0.789 to 0.805 and fibre 0.698 to 0.770; the pooled number collapses because the trees rank
fibre customers too low to separate them from non-fibre ones. Mean predicted churn on the 461
fibre holdout rows was 0.083 for plain gradient boosting and 0.139 for the balanced variant,
against an actual 0.434; the linear model predicted 0.229. A tree cannot split on a one-hot
column that is constant zero in training, and 196 holdout rows sit above the largest
`MonthlyCharges` the training set contains, where a tree's last bin flattens and a linear
coefficient keeps extrapolating.

The fix is one line of ordering: holdout, then val, then the future batch, then train. The
holdout is unchanged, the same 1,057 customers before and after, because it is still drawn
first. Train still contains no fibre customer, so Session 3's drift story is intact; the future
batch shrinks from 2,635 to 2,114 rows. Val now matches the holdout on both things that matter,
churn rate 0.2654 against 0.2649 and fibre share 0.435 against 0.436, and the correlation
between validation and holdout PR-AUC across the twelve configs measured at the time rose to
0.97. `tests/test_split.py::test_val_mirrors_the_holdout_population` is the regression test.

## 2026-09-05 Gate thresholds re-derived, tolerance rule unchanged
The first `configs/thresholds.yaml` (commit 10355c4) applied a fixed tolerance rule to run
ae4f0cd88d4b4d439bbcadb545bd01b6, and was committed before any Session 2 sweep existed so it
could not be fitted to a result. Fixing the split changed which rows a model trains on, so that
baseline stopped being a like-for-like comparison and the same rule was re-applied to the same
incumbent configuration re-measured on the corrected split, run
6903c40b712c4f288f959a48b1aa6589. The rule did not change: 0.02 absolute for the two AUCs, 0.05
absolute for recall at precision 0.5, 20 % relative for Brier. The baseline is the incumbent,
never a sweep winner, and both versions of the file are in git history. Exploratory numbers had
already been seen when the file was rewritten, which is stated in the file itself rather than
glossed over.

## 2026-09-05 Sweep: explicit configs, two families, no optimisation library
`configs/sweep.yaml` lists all 14 configs in full. No grid expansion in code, no Optuna or
scikit-learn `GridSearchCV`: the file is the experiment record, and a diff shows exactly which
hyperparameter moved. Eight logistic regressions over C and `class_weight`, six histogram
gradient boosting models over learning rate, leaf count and L2. Both families run through the
same preprocessor, so every child carries the same feature-code hash and the same 45 columns
and the runs are comparable. The incumbent `C=1.0` is in the grid, enforced by
`tests/test_configs.py::test_the_sweep_contains_the_incumbent`: a sweep that cannot reproduce
the champion it is trying to replace is not a regression check. It was missing from the first
draft.

## 2026-09-05 Selection on val, one look at the holdout, enforced by an object
Selection is PR-AUC on val. The holdout is handed out by `split.HoldoutBudget`, which allows a
fixed number of looks, records the reason for each, and raises on the next one. The sweep gives
it a budget of one, spends it on the winner after selection is over, and calls
`assert_fully_spent()`. Selection with a per-candidate holdout score would report the maximum of
14 samples rather than an estimate; this is where that stops being a promise. Reading holdout
customerIDs for the leak check in `fit_model` is not scoring and does not spend the budget. The
eval gate scores the holdout on purpose, in a separate command: the sweep may not use it to
choose, while the gate exists to measure the one model already chosen.

## 2026-09-05 Promotion: sweep winner takes champion, outgoing keeps its version as previous
`register.promote_to_champion` points `champion` at the new version and moves the outgoing one
to the alias `previous`. Nothing is deleted, so a rollback is an alias move and Session 3's
challenger comparison has something to read. Session 2 promotes the val-PR-AUC winner without a
head-to-head test; the eval gate is what stops a regression going anywhere. The head-to-head
rule that refuses a challenger which does not beat the champion on the fixed holdout is Session
3. Each sweep writes `promotion.json` recording both versions, and the outgoing model's holdout
numbers are read back from its own run rather than re-scored, which would have been a second
look at the holdout.

## 2026-09-05 Only the winner's model is stored, in the run that trained it
`mlflow.sklearn.log_model` costs about 20 seconds. Logging all 14 would be most of the sweep's
runtime for artifacts nobody loads. The winning child run is resumed and the model is logged
there, so the registry version points at the run holding the params and metrics that produced
it, rather than at the parent. The sweep is seeded and finishes in 26 seconds, so re-fitting a
losing config is cheaper than storing every pickle, and each child already carries the params,
the dataset digest and the feature-code hash that reproduce it.

## 2026-09-05 CI builds the registry before it runs the gate
`mlflow.db` and `mlruns/` are gitignored, so a runner starts with no registry and
`python -m mlops_loop eval` alone would have nothing to score. The `gate` job runs
`train` then `eval`, which means every push reproduces the sweep from the dataset URL and
gates the champion it just built. It also makes CI a standing check that the pipeline still
runs end to end from a clean clone. `pytest` runs as a separate parallel job so a test failure
and a gate failure are distinguishable at a glance. Weekly schedule added: the dataset is
fetched from a URL that can change, and the pipeline is deterministic, so a red Monday means
the world moved rather than the code.

## 2026-09-05 Shared ingest, validate and split path
`mlops_loop/dataset.py` holds steps 1 to 3 as one call used by `skeleton`, `train` and `eval`.
Three commands rebuilding splits from three copies of the same code is how they drift apart, and
the gate must score the exact rows the sweep's winner was measured on. Rebuilding is cheap and
seeded, so it is preferred to trusting `data/splits/*.parquet`, which is gitignored and absent
on a fresh runner.
