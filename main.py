"""Unified command-line entry point for the Rumor Detection System."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys

from src.cli.arguments import port_number, positive_int


DEFAULT_CONFIG = "configs/bertweet.yaml"


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        dest="command_config",
        default=None,
        help=f"Path to model/training config YAML. Defaults to {DEFAULT_CONFIG}.",
    )


def _config_path(args: argparse.Namespace) -> str:
    return args.command_config or args.config


def _load_env_file(path: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path)


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from src.model import configure_inference
    from src.web.app import create_app

    if args.reload:
        reload_settings = {
            "RUMOR_CONFIG_PATH": _config_path(args),
            "RUMOR_CHECKPOINT_PATH": args.checkpoint,
            "RUMOR_DEVICE": args.device,
            "RUMOR_FORCE_MOCK": "1" if args.mock_model else "0",
            "RUMOR_ENABLE_RAG": "0" if args.no_rag else "1",
            "RUMOR_ENABLE_LLM": "0" if args.no_llm else "1",
        }
        for name, value in reload_settings.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        uvicorn.run(
            "src.web.app:create_runtime_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    configure_inference(
        config_path=_config_path(args),
        checkpoint=args.checkpoint,
        device=args.device,
        force_mock=args.mock_model,
    )
    app = create_app(enable_rag=not args.no_rag, enable_llm=not args.no_llm)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def train_model(args: argparse.Namespace) -> None:
    from src.cli.train import run

    run(
        _config_path(args),
        args.device,
        args.epochs,
        args.output_dir,
        args.checkpoint_dir,
    )


def evaluate_model(args: argparse.Namespace) -> None:
    from src.cli.evaluate import run

    run(_config_path(args), args.test, args.checkpoint, args.output, args.device)


def predict_text(args: argparse.Namespace) -> None:
    from src.cli.predict import run

    text = args.text_arg or args.text
    if not text:
        raise SystemExit("predict requires text as a positional argument or --text.")
    run(text, _config_path(args), args.checkpoint, args.device, args.json)


def prepare_data(args: argparse.Namespace) -> None:
    from src.cli.prepare_extra_datasets import run

    run(
        args.input_dir,
        args.output_dir,
        args.shared_task,
        args.gossipcop_output,
        args.shared_task_output,
        args.combined_output,
        args.gossipcop_text_fields,
    )


def plot_training_history(args: argparse.Namespace) -> None:
    from src.cli.plot_history import run

    run(_config_path(args), args.history, args.output_dir)


def rag_query(args: argparse.Namespace) -> int:
    from src.rag import cli as rag_cli
    from src.rag import config as rag_config

    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("rag query requires a statement.")
    if args.top_k is not None:
        rag_config.TOP_K = args.top_k

    if args.json:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = rag_cli.run_once(query)
        print(json.dumps(result, ensure_ascii=False))
    else:
        result = rag_cli.run_once(query)
        print("\n========== Final Result (JSON) ==========")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("label") == "unavailable" else 0


def _normalize_legacy_args(argv: list[str]) -> list[str]:
    if not argv:
        return ["serve"]
    if "--train" in argv:
        index = argv.index("--train")
        return [*argv[:index], "train", *argv[index + 1 :]]
    if "--display" in argv:
        index = argv.index("--display")
        return [*argv[:index], "serve", *argv[index + 1 :]]
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rumor Detection System")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Default config YAML path.")
    parser.add_argument("--env-file", default=".env", help="Environment file to load.")
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        default="INFO",
        help="Python logging level.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose tracebacks.")

    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI web demo.")
    _add_config_arg(serve_parser)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=port_number, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument("--checkpoint", default=None)
    serve_parser.add_argument("--device", default=None)
    serve_parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval.")
    serve_parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls.")
    serve_parser.add_argument("--mock-model", action="store_true", help="Force mock classifier.")
    serve_parser.set_defaults(func=serve)

    train_parser = subparsers.add_parser("train", help="Train the BERTweet model.")
    _add_config_arg(train_parser)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--epochs", type=positive_int, default=None)
    train_parser.add_argument("--output-dir", default=None)
    train_parser.add_argument("--checkpoint-dir", default=None)
    train_parser.set_defaults(func=train_model)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    _add_config_arg(eval_parser)
    eval_parser.add_argument("--test", default=None, help="Evaluate a labeled CSV file.")
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--device", default=None)
    eval_parser.add_argument("--output", default=None)
    eval_parser.set_defaults(func=evaluate_model)

    predict_parser = subparsers.add_parser("predict", help="Predict one statement.")
    _add_config_arg(predict_parser)
    predict_parser.add_argument("text_arg", nargs="?", default=None)
    predict_parser.add_argument("--text", default=None)
    predict_parser.add_argument("--checkpoint", default=None)
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--json", action="store_true")
    predict_parser.set_defaults(func=predict_text)

    data_parser = subparsers.add_parser("prepare-data", help="Convert extra public datasets.")
    data_parser.add_argument("--input-dir", default="datasets/raw_data")
    data_parser.add_argument("--output-dir", default="datasets")
    data_parser.add_argument("--shared-task", default="datasets/raw_data/shared_task_dev.jsonl")
    data_parser.add_argument("--gossipcop-output", default="gossipcop_extra.csv")
    data_parser.add_argument("--shared-task-output", default="shared_task_extra.csv")
    data_parser.add_argument("--combined-output", default="all_extra.csv")
    data_parser.add_argument(
        "--gossipcop-text-fields",
        nargs="+",
        default=["title", "description", "text"],
        help="GossipCop text fields to concatenate.",
    )
    data_parser.set_defaults(func=prepare_data)

    plot_parser = subparsers.add_parser("plot-history", help="Plot training curves.")
    _add_config_arg(plot_parser)
    plot_parser.add_argument("--history", default=None)
    plot_parser.add_argument("--output-dir", default=None)
    plot_parser.set_defaults(func=plot_training_history)

    rag_parser = subparsers.add_parser("rag", help="RAG utilities.")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", required=True)
    rag_query_parser = rag_subparsers.add_parser("query", help="Run a standalone RAG query.")
    rag_query_parser.add_argument("query", nargs="+")
    rag_query_parser.add_argument("--top-k", type=positive_int, default=None)
    rag_query_parser.add_argument("--json", action="store_true")
    rag_query_parser.set_defaults(func=rag_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = _normalize_legacy_args(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)
    if not hasattr(args, "func"):
        args = parser.parse_args([*normalized_argv, "serve"])
    logging.basicConfig(level=getattr(logging, args.log_level))
    _load_env_file(args.env_file)

    try:
        result = args.func(args)
    except Exception:
        if args.debug:
            raise
        logging.exception("Command failed")
        return 1
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
