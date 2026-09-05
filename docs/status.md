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
pull request.

Green: https://github.com/GabrieleBosi/mlops-loop/actions/runs/33947087136 at commit `4bf3855`.

The first run failed, and it was a real bug rather than a flake:
https://github.com/GabrieleBosi/mlops-loop/actions/runs/33946866722. CI calls `pytest -q`, which
does not put the working directory on `sys.path`, while every local check had used
`python -m pytest`, which does. All four test modules failed to import `mlops_loop`. Fixed by
`pythonpath = ["."]` in `pyproject.toml`, and the suite is now checked with the bare `pytest`
command the README documents and CI runs. Note for later sessions: run the command CI runs, not a
convenient equivalent.

### Repository
https://github.com/GabrieleBosi/mlops-loop, public, MIT. `.env`, `data/`, `mlruns/` and `mlflow.db`
have been gitignored since the first commit; the tracked tree was grepped for `sk-ant`, `API_KEY=`,
`ghp_`, `github_pat_`, AWS key ids and PEM private-key headers before it was pushed, and none is
present. 36 files tracked, no dataset and no MLflow store among them.

## 2026-09-05, end of Session 2

### Exists
Steps 1 to 8 and step 11 of the decomposition table. `python -m mlops_loop train` runs the sweep
and promotes; `python -m mlops_loop eval` gates the champion and exits 1 below threshold. `drift`
and `reproduce` still print which session builds them and exit 2.

The sweep: `configs/sweep.yaml` lists 14 configs in full, 8 logistic regressions over C and
`class_weight` and 6 histogram gradient boosting models over learning rate, leaf count and L2. No
grid expansion in code and no optimisation library. Each config is a child run under one parent,
carrying its params, the four validation metrics, its calibration curve, the dataset digest and
the feature-code hash.

Sweep parent run `4afa568e01354c78acc720694a94a99c`, commit `be86fd5`, 14 configs in 25.7 seconds,
well inside the three-minute budget. Winner `lr-c0.05-balanced`, child run
`12c545f782044a27822e6b3d83f5c44a`, val PR-AUC 0.5851. Registered as `churn` version 2 with the
`champion` alias; version 1, run `6903c40b712c4f288f959a48b1aa6589`, keeps its version and holds
`previous`. The full ranked table with every child run id is in `reports/sweep_comparison.md`.

The holdout was looked at exactly once, by the winner, after selection was over. That is enforced
by `split.HoldoutBudget`, which hands the frame out a fixed number of times, records the reason
for each, raises on the next, and is asserted fully spent before the sweep returns.

Gate run `de2de7f753cc45ea87d42752d2b32d9c`, champion version 2, 1,057 holdout rows:

| metric | value | threshold | margin | result |
|--------|------:|-----------|-------:|--------|
| `holdout_roc_auc` | 0.8014 | >= 0.7699 | +0.0315 | pass |
| `holdout_pr_auc` | 0.5944 | >= 0.5164 | +0.0780 | pass |
| `holdout_recall_at_precision_50` | 0.6893 | >= 0.5964 | +0.0929 | pass |
| `holdout_brier` | 0.1962 | <= 0.1981 | +0.0019 | pass |

The Brier margin is two thousandths. Selection on PR-AUC alone rewards ranking and ignores whether
the probabilities mean anything, so it picked the reweighted model over the otherwise identical
`lr-c0.05`, which ranks marginally worse and calibrates much better, 0.1604 against 0.1868 on val.
The gate nearly caught the champion the sweep chose. Putting calibration into the selection rule
is the obvious next move and is not done.

63 tests pass locally in about 66 seconds, and identically on ubuntu in CI.

### What Session 2 found and fixed
The first sweep selected `hgb-lr0.05-leaf15-balanced` on val ROC-AUC 0.8342, and it scored 0.6281
on the holdout. Error analysis attributed it to step 3, Split: val was drawn from the reference
pool, which contains no fibre-optic customer, while the holdout is 43.6 % fibre, so validation
ranked candidates on a sub-population the champion does not serve. On holdout slices every
candidate was comparable; the pooled number collapsed because the trees scored the 461 fibre
customers far too low, mean predicted churn 0.083 to 0.139 against an actual 0.434, while the
linear model predicted 0.229. A tree cannot split on a one-hot column that is constant zero in
training, and 196 holdout rows sit above the largest `MonthlyCharges` in the training set, where a
tree's last bin flattens.

Fixed by reordering the split to holdout, val, future batch, train. The holdout is unchanged, the
same 1,057 customers, because it is still drawn first. Train still contains no fibre customer, so
Session 3's drift story is intact; the future batch went from 2,635 to 2,114 rows. Correlation
between validation and holdout PR-AUC across the configs rose to 0.97.
`tests/test_split.py::test_val_mirrors_the_holdout_population` is the regression test. A second
defect, mine: the sweep grid did not contain `C=1.0`, the incumbent, so it could not have been a
regression check on the champion. Added, and enforced by
`tests/test_configs.py::test_the_sweep_contains_the_incumbent`.

### Reproducibility
CI, on ubuntu with an empty registry, produced the same winner and the same four holdout metrics
to four decimal places as the Windows laptop: 0.8014, 0.5944, 0.6893, 0.1962. Different run ids
and a version 1 rather than 2, because the runner starts from nothing, but identical numbers.

### Does not exist
- Promotion on merit. The sweep winner takes `champion` with no head-to-head test against the
  current champion on the fixed holdout. `promotion.json` records both sides of every promotion so
  Session 3 can turn it into a rule.
- Calibration in the selection rule, as argued above.
- `drift`: PSI per feature and on the prediction distribution, the retrain trigger, the
  champion-versus-challenger comparison.
- `reproduce`, `reports/error_analysis.md`, `docs/preflight.md`.
- Track B, the GenAI evaluation suite in order-processing-workflow.

### CI
`.github/workflows/ci.yml` runs two parallel jobs on ubuntu-latest with Python 3.12, on push, on
pull request, and weekly on Mondays at 04:17 UTC. `pytest` runs the suite. `sweep and eval gate`
runs `python -m mlops_loop train` then `python -m mlops_loop eval`. The runner starts with no
registry, because `mlflow.db` and `mlruns/` are gitignored, so the gate scores a champion this
commit built from the dataset URL and nothing else. Two jobs rather than one so a test failure and
a gate failure are distinguishable at a glance.

Green: https://github.com/GabrieleBosi/mlops-loop/actions/runs/33948414125 at commit `3d6311d`,
both jobs.

Red on purpose, as the definition of done requires:
https://github.com/GabrieleBosi/mlops-loop/actions/runs/33948528226 at commit `47098aa`, on the
branch `gate-fails-on-purpose`, since deleted. `holdout_roc_auc` was raised from 0.7699 to 0.95,
which nothing this repo can train will reach. `pytest` stayed green and `sweep and eval gate` went
red, so the signal points at the gate rather than at the code:

```
  metric                                 value   threshold    margin   result
  holdout_roc_auc                       0.8014    >=0.9500   -0.1486   FAIL
  holdout_pr_auc                        0.5944    >=0.5164   +0.0780   pass
  holdout_recall_at_precision_50        0.6893    >=0.5964   +0.0929   pass
  holdout_brier                         0.1962    <=0.1981   +0.0019   pass

  gate FAILED on 1 check(s): holdout_roc_auc
##[error]Process completed with exit code 1.
```

Locally the same change gave exit code 1 and gate run `dfc075fbaae349439cfd5c2b8511769d`.

## 2026-09-05, end of Session 3

### Exists
All eleven steps of the decomposition table. `drift`, `reports` and `reproduce` are built, so no
command in `CLAUDE.md` prints "not built" any more.

Everything below came from one `python -m mlops_loop reproduce` at commit `0b1cd3a`, 26 logged
runs, 63.5 seconds total on a Windows laptop:

| stage | seconds | result |
|-------|--------:|--------|
| skeleton | 22.0 | run `74e4cde1680b4a78ad06d270e7c53fd6`, version 1 |
| train | 19.7 | parent `a15a64efaaf94f8dbbe980cec4ea13f3`, 14 configs, version 2 |
| eval | 2.3 | run `07797376000849eab75e3137d91fe62e`, 4 of 4 passed |
| drift | 14.8 | 2 batches, 2 decisions, 1 promotion |
| reports | 2.6 | run `ac4850f895b5402e898cc31e97588504` |
| eval (final champion) | 2.1 | run `ea66c83c2d454c329c46145c48340c4a`, version 3, passed |

Registry: `churn` versions 1, 2 and 3. Version 3 (`challenger-fibre-monthly`) holds `champion`,
version 2 (`lr-c0.05-balanced`) holds `previous`.

### Drift
`configs/drift.yaml` defines two future batches as feature slices, a PSI threshold for each of the
19 model features, a tighter one for the prediction distribution, and the promotion margin. Both
batches partition the fibre pool Session 1 reserved, so train, val and holdout did not move.

| batch | rows | churn | monitor run | breaches | prediction PSI |
|-------|-----:|------:|-------------|---------:|---------------:|
| `fibre-monthly` | 1,451 | 0.5410 | `c934991a348d4126a48cf7301d99296d` | 16 of 20 | 4.3735 |
| `fibre-committed` | 663 | 0.1207 | `cfa00a696f894394b432a60020c9288b` | 16 of 20 | 0.9595 |

Loudest on both: `InternetService` 26.97, `MonthlyCharges` 10.67 and 11.36, `Contract` 7.46 and
6.04. Same input drift, opposite label drift, which is why a PSI breach only earns the right to
train a challenger.

### The two promotion decisions
Rule, fixed in `configs/drift.yaml` before either batch ran: promote only if PR-AUC on the fixed
holdout improves by more than 0.01, both models scored in the same run on the same 1,057 rows.

`fibre-monthly`, decision run `853f20f3e0d64adab9ea9158ca970bb6`, **promoted**. Challenger
`5873c3dcee694a6a9b32316f882662b9`, the champion's own config on 4,125 rows. PR-AUC 0.5944 to
0.6515 (+0.0571), ROC-AUC +0.0244, recall at precision 0.5 +0.0857, Brier 0.1962 to 0.1643.
Registered as version 3; version 2 took `previous`.

`fibre-committed`, decision run `7d1ee8ce585d4e05ac66fc3bfcd387b1`, **rejected**. Challenger
`1039e97ee6fa43a087271e30d4cfa583` on 3,337 rows. PR-AUC 0.6515 to 0.6500 (−0.0015), so under the
margin, even though ROC-AUC improved +0.0076 and recall +0.0250. Brier got much worse, 0.1643 to
0.2272. Neither outcome was engineered.

### Gate
Final gate `ea66c83c2d454c329c46145c48340c4a` on version 3, 4 of 4 passed, and with far more room
than Session 2's champion had: ROC-AUC 0.8258 (+0.0559), PR-AUC 0.6515 (+0.1351), recall at
precision 0.5 0.7750 (+0.1786), Brier 0.1643 (+0.0338 against +0.0019 for version 2). Retraining
on the drifted batch improved calibration as a side effect.

### Error analysis
`reports/error_analysis.md`, run `ac4850f895b5402e898cc31e97588504`, champion version 3. 254 of
1,057 holdout rows wrong at threshold 0.5 (24.0 %): 72 false negatives, 182 false positives.
Attribution is code, in `mlops_loop/analysis.py`.

| component | sampled (20) | all (254) |
|-----------|-------------:|----------:|
| split | 0 | 1 |
| features | 2 | 40 |
| **model** | **17** | **188** |
| threshold | 1 | 25 |

Biggest cell `model`. 140 of those 188 are false positives at a median predicted probability of
0.739, and the champion flags 36.9 % of the holdout against an actual 26.5 %. Named cheapest fix:
add Brier as a constraint to the sweep's selection rule and re-run `train`, two lines of config
and 26 seconds. Not applied, by design.

### Reproduce from a clean clone
`git clone` into an empty directory, `py -3.12 -m venv`, `pip install -r requirements.txt`: **101
seconds**. Then `python -m mlops_loop reproduce`: **84 seconds** wall clock, exit 0. Total **3
minutes 5 seconds**, against the brief's budget of 10. Every metric came back identical to four
decimal places: gate 0.8258 / 0.6515 / 0.7750 / 0.1643, tally 0/2/17/1 sampled and 1/40/188/25
overall, `fibre-monthly` +0.0571 promoted and `fibre-committed` −0.0015 rejected. Only the run ids
differ, because they are new runs.

One note for anyone repeating this on Windows: the clone has to sit in a short path. mlflow ships
migration files nested deeply enough that a long temporary directory hits the 260-character limit
and `pip install` fails partway through.

### Tests and CI
109 tests, about 2 minutes 11 seconds locally. PSI is checked against two hand-computed values on
both the categorical and the numeric path. `tests/test_retrain.py` asserts the fixed holdout is
never trained on, that every decision run carries both run ids and the margin, and that the
verdict follows the rule rather than a stored answer.

CI now runs `python -m mlops_loop reproduce` in place of `train` then `eval`, so every push
rebuilds the registry from nothing and gates the champion it just built.

### Does not exist
- Promotion on merit inside the sweep. `train` still gives `champion` to the sweep winner with no
  head-to-head test against the incumbent. `drift` does the comparison properly, and
  `promotion.json` records both sides, so the rule needs moving rather than writing.
- Calibration in the selection rule, which the error analysis argues for and Session 2 flagged
  independently when version 2 cleared the Brier line by 0.0019.
- A component-level test that fails when calibration degrades. `docs/preflight.md` item 9 is
  unticked for this reason.
- Track B, the GenAI evaluation suite in order-processing-workflow. Session 4.

`docs/preflight.md` is filled in: seven of ten items ticked with evidence links, three left
unticked with a line each on why.
