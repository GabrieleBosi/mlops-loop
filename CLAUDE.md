# mlops-loop

One tabular ML system run through the full lifecycle on MLflow: tracked experiments, a model registry with champion and challenger promotion, drift-triggered retraining, a serving endpoint that loads from the registry, and a CI gate that fails closed. A second track logs a GenAI evaluation suite (regression, safety, multilingual) into MLflow with traces; it lives in the order-processing-workflow repo and is linked from the README.

## Read first
- docs/brief.md: scope, decomposition table, phase plan, definition of done
- docs/status.md: what exists and what does not, updated at the end of every session
- docs/reference/field-guide-rules.md: the engineering rules this repo follows. Do not open the .html files in docs/reference; the .md distils them and they cost context.

## Rules that override defaults
- Lowest autonomy that passes the evals. The classical pipeline is a fixed workflow: plain Python, no LLM, no agent framework. Every step is deterministic code.
- Skeleton first. The pipeline must run end to end on real data before any single step is made good. Do not polish components in isolation.
- Every step logs to MLflow: params, metrics, dataset digest, feature-code hash, git commit, artifacts. If it is not in a run, it did not happen.
- Evals are the gate. `python -m mlops_loop eval` runs in one command and exits non-zero below the thresholds in configs/thresholds.yaml. CI runs it on every push.
- Error analysis before optimisation. Failures are attributed to the first failing component in reports/error_analysis.md before any fix is chosen.
- Fail closed. Missing data, a schema violation, a metric that cannot be computed: the pipeline stops and says why. Never pass on partial results.
- The fixed holdout is scored, never trained on. Assert it in code.
- Least privilege. Secrets only from environment variables. .env is gitignored from the first commit. No key is ever written to a file, a log, or an MLflow tag.
- No fabricated numbers. Every figure in the README comes from a logged run and cites the run id.
- No new dependency without a dated line in docs/decisions.md saying what it replaces and why.

## Stack (verify versions and APIs against current docs on first use)
- Python 3.12, virtualenv at .venv, pip with requirements.txt pinned
- mlflow 3.x (registry uses aliases, not stages), scikit-learn, pandas, pyarrow, pandera, numpy, fastapi, uvicorn, pytest, pyyaml
- MLflow backend: sqlite:///mlflow.db, artifacts under ./mlruns, both gitignored
- Serving: FastAPI loading models:/churn@champion
- CI: GitHub Actions, ubuntu-latest, Python 3.12
- No make: the CLI is `python -m mlops_loop <command>` and CI calls the same commands

## Commands
python -m mlops_loop skeleton    # steps 1 to 8 once, one model, everything logged
python -m mlops_loop train       # tracked sweep over configs/sweep.yaml
python -m mlops_loop eval        # gate against configs/thresholds.yaml, exit 1 on failure
python -m mlops_loop drift       # monitor the future batch, retrain and challenge if drifted
python -m mlops_loop serve       # uvicorn on :8000
python -m mlops_loop ui          # mlflow ui on :5000
python -m mlops_loop reproduce   # skeleton, train, eval, drift, reports, from a clean clone

## Writing (README, docstrings, commit messages, reports)
- Plain English, short sentences, no em dashes, no marketing adjectives.
- Say what was measured, on what data, in which run. Never "robust", "seamless", "production-grade", "state of the art".
- Commit messages: imperative, one line, what and why.

## Working with Claude Code
- One session per phase. Run /clear between phases; this file, the brief and status.md carry the state.
- Before writing code in a phase, restate that phase's definition of done from docs/brief.md and list the files you will touch.
- After each phase: run pytest and the eval gate, paste the summary with run ids, update docs/status.md and docs/decisions.md. Do not claim anything you did not run.
