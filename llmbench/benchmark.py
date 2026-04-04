import subprocess
import json
import sys
import tempfile
from pathlib import Path


# Map GGUF filename prefixes to HuggingFace tokenizer repos
TOKENIZER_MAP = {
    "Qwen3.5-": "Qwen/Qwen3.5-4B-Base",
    "Qwen3-": "Qwen/Qwen3-0.6B",
    "Qwen2.5-Coder-": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen2.5-": "Qwen/Qwen2.5-0.5B",
    "gemma-4-": "google/gemma-3-4b-it",
    "gemma-3-": "google/gemma-3-4b-it",
    "Mistral-": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3": "meta-llama/Llama-3.2-1B",
    "Phi-": "microsoft/Phi-3-mini-4k-instruct",
}


def _resolve_tokenizer(repo_id: str, model_name: str, tokenizer: str | None) -> str:
    """Resolve a HuggingFace tokenizer repo.

    Priority:
    1. Explicit --tokenizer flag
    2. Strip -GGUF from the GGUF repo ID (e.g. Jackrong/Foo-GGUF -> Jackrong/Foo)
    3. Filename prefix map as last resort
    """
    if tokenizer:
        return tokenizer
    # Most GGUF repos have a non-GGUF sibling with the tokenizer
    if repo_id.endswith("-GGUF"):
        return repo_id.removesuffix("-GGUF")
    for prefix, repo in TOKENIZER_MAP.items():
        if model_name.startswith(prefix):
            return repo
    raise ValueError(
        f"Cannot auto-detect tokenizer for '{model_name}' (repo: {repo_id}). "
        f"Pass --tokenizer with a HuggingFace repo ID."
    )


# Primary metrics to extract per task
PRIMARY_METRICS = {
    "hellaswag": "acc_norm",
    "humaneval": "pass@1",
    "mbpp": "pass@1",
    "mmlu": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "truthfulqa_mc2": "acc",
    "winogrande": "acc",
}


def run_benchmark(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
    repo_id: str,
    tokenizer: str | None = None,
) -> dict:
    """Run lm-evaluation-harness against the local llama-server endpoint.

    Tries the Python API first, falls back to CLI subprocess.
    Returns {task_name: {metric: score, ...}, ...}.
    """
    tokenizer_repo = _resolve_tokenizer(repo_id, model_name, tokenizer)
    print(f"Using tokenizer: {tokenizer_repo}")
    try:
        return _run_via_library(port, tasks, limit, model_name, tokenizer_repo)
    except ImportError as e:
        print(f"Library invocation failed ({e}), falling back to CLI...")
        return _run_via_cli(port, tasks, limit, model_name, tokenizer_repo)


def _run_via_library(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
    tokenizer_repo: str,
) -> dict:
    """Use lm_eval.simple_evaluate() directly."""
    import lm_eval

    model_args = (
        f"model={model_name},"
        f"base_url=http://localhost:{port}/v1/completions,"
        f"tokenizer_backend=huggingface,"
        f"tokenizer={tokenizer_repo},"
        f"num_concurrent=4,"
        f"timeout=600"
    )

    print(f"Running benchmark: tasks={tasks}, limit={limit}")
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=tasks,
        limit=limit,
        log_samples=False,
    )

    return _extract_scores(results["results"])


def _run_via_cli(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
    tokenizer_repo: str,
) -> dict:
    """Fall back to lm_eval CLI and parse JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            "--model_args", (
                f"model={model_name},"
                f"base_url=http://localhost:{port}/v1/completions,"
                f"tokenizer_backend=huggingface,"
                f"tokenizer={tokenizer_repo},"
                f"num_concurrent=4,"
                f"timeout=600"
            ),
            "--tasks", ",".join(tasks),
            "--limit", str(limit),
            "--output_path", tmpdir,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Benchmark failed:\n{result.stderr}")

        # Find the results JSON file
        results_files = list(Path(tmpdir).rglob("results.json"))
        if not results_files:
            raise RuntimeError(f"No results.json found in {tmpdir}\nstdout: {result.stdout[-500:]}")

        with open(results_files[0]) as f:
            data = json.load(f)

        return _extract_scores(data["results"])


def _extract_scores(results: dict) -> dict:
    """Extract primary metrics from lm-eval results dict."""
    scores = {}
    for task_name, task_results in results.items():
        scores[task_name] = {}
        # Try the known primary metric first
        primary = PRIMARY_METRICS.get(task_name)
        for key, value in task_results.items():
            if key.startswith("alias"):
                continue
            # Include the primary metric and any stderr
            if isinstance(value, (int, float)):
                scores[task_name][key] = round(value, 4)

    return scores
