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
