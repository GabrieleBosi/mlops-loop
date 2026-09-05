"""python -m mlops_loop <command>. CI calls the same commands as a person does."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import tracking

NOT_BUILT = {
    "drift": ("Session 3", "PSI on the future batch, retrain and challenge"),
    "reproduce": ("Session 3", "skeleton, train, eval, drift and reports end to end"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mlops_loop",
        description="One tabular churn model through the full MLflow lifecycle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    skeleton = subparsers.add_parser("skeleton", help="steps 1 to 8 once, one model, all logged")
    skeleton.add_argument("--config", default=None, help="path to skeleton.yaml")
    skeleton.add_argument("--source", default=None, help="override the dataset URL with a path")

    train = subparsers.add_parser("train", help="tracked sweep over configs/sweep.yaml")
    train.add_argument("--config", default=None, help="path to skeleton.yaml")
    train.add_argument("--sweep", default=None, help="path to sweep.yaml")
    train.add_argument("--source", default=None, help="override the dataset URL with a path")

    evaluate = subparsers.add_parser(
        "eval", help="gate the champion against configs/thresholds.yaml, exit 1 on failure"
    )
    evaluate.add_argument("--config", default=None, help="path to skeleton.yaml")
    evaluate.add_argument("--thresholds", default=None, help="path to thresholds.yaml")
    evaluate.add_argument("--source", default=None, help="override the dataset URL with a path")

    serve = subparsers.add_parser("serve", help="FastAPI on :8000 serving models:/churn@champion")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ui = subparsers.add_parser("ui", help="mlflow ui on :5000 against the local store")
    ui.add_argument("--port", type=int, default=5000)

    for name, (session, what) in NOT_BUILT.items():
        subparsers.add_parser(name, help=f"not built until {session}: {what}")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command in NOT_BUILT:
        session, what = NOT_BUILT[args.command]
        print(
            f"'{args.command}' is not built yet. {session} adds it: {what}.\n"
            f"See docs/status.md for what exists today.",
            file=sys.stderr,
        )
        return 2

    if args.command == "skeleton":
        from . import skeleton

        result = skeleton.run(config_path=args.config, source=args.source)
        print(skeleton.format_result(result))
        return 0

    if args.command == "train":
        from . import train

        result = train.run(config_path=args.config, sweep_path=args.sweep, source=args.source)
        print(train.format_result(result))
        return 0

    if args.command == "eval":
        from . import gate

        try:
            result = gate.run(
                config_path=args.config, thresholds_path=args.thresholds, source=args.source
            )
        except gate.GateError as exc:
            print(f"eval gate could not run: {exc}", file=sys.stderr)
            return 1
        print(gate.format_result(result))
        return 0 if result.passed else 1

    if args.command == "serve":
        import uvicorn

        uvicorn.run("mlops_loop.serve:app", host=args.host, port=args.port)
        return 0

    if args.command == "ui":
        from . import config

        config.load_dotenv()
        uri = os.environ.get("MLFLOW_TRACKING_URI", tracking.DEFAULT_TRACKING_URI)
        return subprocess.call(
            [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri,
             "--port", str(args.port)]
        )

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
