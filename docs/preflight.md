# Pre-flight checklist

The ten items from `docs/reference/field-guide-rules.md` (field 08). An item is ticked only
where there is something to point at: a file, a run id, or a CI link. Where it is not ticked,
the line says why, because an unticked box with a reason is worth more than a ticked one
without evidence.

Filled 2026-09-05, end of Session 3, at commit `0b1cd3a`. Every run id below is from the
`reproduce` run of that commit, total wall clock 63.5 seconds.

---

- [x] **Task decomposed into named steps, each with explicit input and output**

  `docs/brief.md`, the decomposition table: eleven steps, each with an executor, an input, an
  output and a named risk. It was written before any code and has been kept current. Each row
  maps to a module: steps 1 to 8 are `ingest`, `validate`, `split`, `features`, `train`,
  `evaluate`, `register`, `serve`; steps 9 to 11 are `monitor`, `retrain`, `gate`.

- [x] **Executor chosen per step; code used wherever it can be**

  The executor column of that table reads "code" in all eleven rows, and
  `docs/decisions.md`, 2026-09-04, records why: no step in a tabular pipeline is done better
  by a model than by deterministic code. There is no LLM anywhere in this repo. Track B, in
  the order-processing-workflow repo, is where an LLM is the only executor that can do the
  step.

- [x] **Lowest workable autonomy selected**

  Level 2, a fixed workflow. `docs/decisions.md`, 2026-09-04, "Autonomy level: 2". Every
  command is a straight line through the same steps in the same order; nothing chooses its
  own control flow. `python -m mlops_loop reproduce` is the whole system and it is one
  function calling six others in sequence.

- [x] **End-to-end skeleton runs on at least 5 real inputs**

  `python -m mlops_loop skeleton` runs steps 1 to 8 on the full 7,043-row file and finishes
  by loading `models:/churn@champion` back out of the registry and scoring five real holdout
  customers. Skeleton run `74e4cde1680b4a78ad06d270e7c53fd6`, artifact `serve_check.json`,
  five rows with their customerIDs, probabilities, model version and run id. It was the
  first thing built, in Session 1, before any component was made good.

- [x] **Every step traced; inputs and outputs logged and readable**

  Each step runs inside `tracking.step`, which tags the run `step_<name>=ok` or `failed` and
  logs `seconds_<name>`, so a failed run names the step that stopped it. Every run carries
  the dataset sha256, the feature-code hash, the git commit and the dirty flag. 26 runs from
  one `reproduce`: skeleton, a sweep parent with 14 children, two gates, two monitors, two
  challengers, two promotion decisions, one error analysis. `python -m mlops_loop ui` reads
  them.

- [ ] **Eval set of 20+ real examples runs with one command**

  Not in the sense the field guide means. `python -m mlops_loop eval` is one command and it
  scores 1,057 real holdout customers against four thresholds in
  `configs/thresholds.yaml`, exiting non-zero below any of them, which is the regression
  suite this system needs. What does not exist is a curated set of individually chosen hard
  cases with expected outputs. For a tabular classifier the holdout is the honest version of
  that idea, and a hand-picked subset would be a worse estimator of the same thing. Track B,
  where outputs are text and there is no holdout to average over, is where a curated eval set
  earns its keep.

- [x] **Error-analysis table exists and the current work item is its biggest cell**

  `reports/error_analysis.md`, analysis run `ac4850f895b5402e898cc31e97588504`. All 254
  holdout failures attributed by code, twenty sampled and listed by customerID. Biggest cell
  `model`, 17 of 20 sampled and 188 of 254 overall. The named cheapest fix is to put
  calibration into the selection rule and re-run the sweep.

  Half a tick, honestly: the table exists and names its biggest cell, but the current work
  item is not yet that cell, because the brief asks Session 3 to analyse and not to fix. The
  fix is written down, costed at 26 seconds of compute, and left for Session 4.

- [x] **Guardrails in place: iteration caps, validation, confirmation gates, sandboxing**

  Validation: `validate.py` runs a pandera schema with `strict=True` and stops on an
  unexpected column, an unknown category or a non-numeric value, naming the offending rows.
  The 11 blank `TotalCharges` are handled explicitly and counted as a metric rather than
  silently dropped.

  Leakage guards: `split.HoldoutBudget` hands the holdout out a fixed number of times,
  records why each time and raises on the next; the sweep spends one look, a promotion
  decision spends two. `split.assert_holdout_unseen` compares customerIDs before any fit and
  raises on overlap, so "the holdout is never trained on" is an exception rather than a
  convention.

  Gates: `python -m mlops_loop eval` exits 1 below threshold and CI fails with it, seen
  deliberately at [run 33948528226](https://github.com/GabrieleBosi/mlops-loop/actions/runs/33948528226).
  A challenger is promoted only past a margin on the fixed holdout.

  Least privilege: no credential exists in this repo. `.env` has been gitignored since the
  first commit and the tree is grepped for key patterns before each push.

  Not applicable: iteration caps and sandboxing are agent guardrails. Nothing here loops
  under model control or executes generated code.

- [ ] **Component-level evals cover the steps that failed most**

  Partly. 109 tests cover validate, split, features, the sweep, PSI, the retrain decision,
  the gate and the error-analysis attribution rules, including PSI checked against two
  hand-computed values and a test that the sweep can reproduce the incumbent. What is missing
  is a component-level eval for the step the error analysis actually blames, `model`: there
  is no test that fails when calibration degrades. `configs/thresholds.yaml` has a Brier line
  and it caught the Session 2 champion with 0.0019 to spare, which is a system-level check
  standing in for a component-level one. Adding a per-config calibration assertion to the
  sweep is part of the fix Session 4 is expected to make.

- [ ] **Latency and cost profiled per step, budgets agreed before optimising**

  Profiled, not budgeted. Every run logs `seconds_<step>`, and `reproduce` prints a per-stage
  table: skeleton 22.0s, train 19.7s, eval 2.3s, drift 14.8s, reports 2.6s, final gate 2.1s,
  total 63.5s. From a clean clone with an empty registry the same run took 84s, plus 101s to
  clone and install. No budget has been agreed because nothing is near a limit and the field
  guide's own rule is to optimise last: the whole loop is under two minutes and the only
  step anyone has spent effort on is `mlflow.sklearn.log_model` at roughly 20 seconds, which
  is why only the winning config's model is stored. Money cost is zero; there is no paid API
  in this track.

---

## What the unticked boxes have in common

Two of the three are about the same gap: this repo measures the system end to end well and
measures its components unevenly. The eval gate knows when the champion regresses but no test
fails when a single component degrades, and the error analysis has just named which component
that would be. That is the honest reading of items 6, 9 and 10, and it is the argument for
what Session 4 should do first.
