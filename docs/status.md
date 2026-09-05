# Status

Updated at the end of every session. Say what exists and what does not.

## 2026-09-04, before Session 1
Nothing built. Brief, rules and CLAUDE.md written. Dataset URL verified.

Not yet: everything in the decomposition table.

## 2026-09-05, end of Session 1

### Exists
Steps 1 to 8 of the decomposition table, as `mlops_loop/{ingest,validate,split,features,train,
evaluate,register,serve}.py`, orchestrated by `skeleton.py` inside one MLflow run.

`python -m mlops_loop skeleton` runs the eight steps on the real 7,043-row file and exits 0.
`python -m mlops_loop serve` starts FastAPI on :8000 and loads `models:/churn@champion` at startup.
`python -m mlops_loop ui` opens the MLflow UI on :5000. `train`, `eval`, `drift` and `reproduce`
print which session builds them and exit 2.

Reference run: `ae4f0cd88d4b4d439bbcadb545bd01b6`, experiment `churn`, commit
`605ab742e0527546959f0cdd40424e7913ede637`, `git_dirty=false`, dataset sha256
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`, feature-code hash
`215d7d44d683a290ba5e4a3561fd2f5aa75b4b960203a84643ad6542a4205b49`, 45 encoded columns.
Registered model `churn` version 1, alias `champion`.

| split | rows | churn rate | ROC-AUC | PR-AUC | recall at precision 0.5 | Brier |
|-------|-----:|-----------:|--------:|-------:|------------------------:|------:|
| val | 671 | 0.1475 | 0.8448 | 0.5072 | 0.5657 | 0.0949 |
| holdout | 1,057 | 0.2649 | 0.8186 | 0.5971 | 0.7536 | 0.1493 |
| future | 2,635 | 0.4163 | 0.7721 | 0.6618 | 0.9572 | 0.2067 |

Train: 2,680 rows, churn rate 0.1466. `total_charges_blank_count` 11, as expected.

The run holds params `dataset_source`, `dataset_sha256`, `feature_code_hash`,
`n_encoded_features`, `split_seed`, `holdout_fraction`, `val_fraction`, `future_rule`,
`model_class` and five `model_*`; metrics `raw_rows`, `total_charges_blank_count`, `rows_*`,
`churn_rate_*`, `seconds_*` per step and the twelve evaluation metrics; tags `git_commit`,
`git_dirty`, `phase`, `model_name`, `model_version`, `model_alias` and `step_*` = ok for all eight
steps; artifacts `calibration_holdout.png`, `split_ids.json`, `feature_columns.json`,
`serve_check.json`, both configs and the model; and the raw dataset as a logged input.

Endpoint checked live against that run on 2026-09-05:
`GET /health` returned `{"status":"ok","model_name":"churn","model_version":"1",
"run_id":"..."}` and `POST /predict` on holdout customer `2316-ESMLS` returned probability
0.1144 with the same version and run id. A body with `Contract: "Three year"` returned 422 naming
the failing check. Those numbers came from the earlier identical run `3d180f5f4b414fdf84034da79fcd3561`
on the same code and data; the reference run above is the one on the committed, clean tree.

29 tests pass locally on Windows in about 33 seconds: unit tests for validate, split and features,
plus one integration test that runs the whole skeleton on the committed 500-row fixture against a
temporary MLflow store and exercises `/health` and `/predict` through `TestClient`. No network.

### Does not exist
- `train`: the tracked sweep over `configs/sweep.yaml`. `configs/sweep.yaml` is not written.
- `eval`: the gate. `configs/thresholds.yaml` is not written. No threshold has been set, which is
  deliberate: setting one after seeing the numbers above would be gaming it.
- Promotion on merit. Session 1 has one model and promotes it unconditionally.
- `drift`: PSI per feature and on the prediction distribution, the retrain trigger, the
  champion-versus-challenger comparison, and the logged promotion decisions.
- `reproduce`, `reports/`, `docs/preflight.md`.
- Track B, the GenAI evaluation suite in order-processing-workflow.
- The gate has not yet been seen to fail on purpose. That evidence link belongs in Session 2.

### CI
Workflow `.github/workflows/ci.yml` runs `pytest -q` on ubuntu-latest with Python 3.12, on push and
pull request. First run: recorded below once the repo is pushed.
