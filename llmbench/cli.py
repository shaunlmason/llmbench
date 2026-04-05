import argparse
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import run_benchmark
from .config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_LIMIT,
    DEFAULT_MODELS_DIR,
    DEFAULT_PORT,
    DEFAULT_TASKS,
    HISTORY_FILE,
)
from .download import ensure_model, parse_model_ref
from .results import print_ranking_table, save_result
from .server import (
    restore_services,
    start_llama_server,
    stop_llama_server,
    stop_services,
    wait_for_health,
)


def cmd_run(args):
    stopped_services = []
    server_process = None
    interrupted = False

    def _stop_server_once():
        nonlocal server_process
        if server_process:
            stop_llama_server(server_process)
            server_process = None

    def _restore_services_once():
        if not args.no_restore and stopped_services:
            restore_services(stopped_services)
            stopped_services.clear()

    def _cleanup(signum=None, frame=None):
        _stop_server_once()
        _restore_services_once()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one benchmark task is required.")

    try:
        for model_ref in args.models:
            repo_id, filename = parse_model_ref(model_ref)

            # 1. Download model if needed
            model_path = ensure_model(repo_id, filename, Path(args.models_dir))

            # 2. Stop services (first iteration only)
            if not stopped_services:
                stopped_services = stop_services(args.gpu)

            # 3. Start llama-server
            server_process = start_llama_server(
                model_path,
                args.gpu,
                args.context_length,
                args.port,
                cache_type_k=args.cache_type_k,
                cache_type_v=args.cache_type_v,
                flash_attn=not args.no_flash_attn,
                jinja=not args.no_jinja,
            )

            # 4. Health check
            if not wait_for_health(args.port, server_process):
                _stop_server_once()
                raise RuntimeError(f"Server failed to start for {model_ref}")

            # 5. Run benchmarks
            print(f"\nBenchmarking {filename} ({args.gpu}, ctx={args.context_length})...")
            scores = run_benchmark(args.port, tasks, args.limit, filename, repo_id, args.tokenizer)

            # 6. Store result
            elapsed = scores.pop("_elapsed_seconds", None)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_ref,
                "gpu_config": args.gpu,
                "context_length": args.context_length,
                "scores": scores,
                "elapsed_seconds": elapsed,
            }
            save_result(entry, Path(args.history_file))
            print(f"Results saved for {filename}")

            # 7. Stop server before next model
            _stop_server_once()

    except KeyboardInterrupt:
        interrupted = True
    finally:
        _stop_server_once()
        _restore_services_once()

    if interrupted:
        print("\nBenchmark interrupted.")
        raise SystemExit(130)

    # 8. Print ranking
    print_ranking_table(Path(args.history_file))


def cmd_results(args):
    print_ranking_table(Path(args.history_file), as_json=args.json)


def main():
    parser = argparse.ArgumentParser(
        prog="llmbench",
        description="Fast local LLM benchmarking for llama-server",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Benchmark one or more models")
    run_parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model references in 'repo_id:filename.gguf' format",
    )
    run_parser.add_argument(
        "--gpu", choices=["gpu0", "gpu1", "both"], default="gpu0",
        help="GPU configuration (default: gpu0)",
    )
    run_parser.add_argument(
        "--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
        help=f"Context window size (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    run_parser.add_argument(
        "--tasks", default=DEFAULT_TASKS,
        help=f"Comma-separated lm-eval tasks (default: {DEFAULT_TASKS})",
    )
    run_parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Samples per task — controls speed (default: {DEFAULT_LIMIT})",
    )
    run_parser.add_argument(
        "--models-dir", default=str(DEFAULT_MODELS_DIR),
        help=f"Directory for GGUF files (default: {DEFAULT_MODELS_DIR})",
    )
    run_parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"llama-server port (default: {DEFAULT_PORT})",
    )
    run_parser.add_argument(
        "--cache-type-k", default="q4_0",
        help="KV cache quantization type for keys (default: q4_0)",
    )
    run_parser.add_argument(
        "--cache-type-v", default="q4_0",
        help="KV cache quantization type for values (default: q4_0)",
    )
    run_parser.add_argument(
        "--no-flash-attn", action="store_true",
        help="Disable flash attention (enabled by default)",
    )
    run_parser.add_argument(
        "--no-jinja", action="store_true",
        help="Disable Jinja template support (enabled by default)",
    )
    run_parser.add_argument(
        "--tokenizer", default=None,
        help="HuggingFace tokenizer repo ID (auto-detected from model name if omitted)",
    )
    run_parser.add_argument(
        "--no-restore", action="store_true",
        help="Don't restart original services after benchmarking",
    )
    run_parser.add_argument(
        "--history-file", default=str(HISTORY_FILE),
        help=f"Path to results history (default: {HISTORY_FILE})",
    )
    run_parser.set_defaults(func=cmd_run)

    # --- results ---
    results_parser = subparsers.add_parser("results", help="View benchmark rankings")
    results_parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of table",
    )
    results_parser.add_argument(
        "--history-file", default=str(HISTORY_FILE),
        help=f"Path to results history (default: {HISTORY_FILE})",
    )
    results_parser.set_defaults(func=cmd_results)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
