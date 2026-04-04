import subprocess
import json
import sys
import tempfile
from pathlib import Path


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
) -> dict:
    """Run lm-evaluation-harness against the local llama-server endpoint.

    Tries the Python API first, falls back to CLI subprocess.
    Returns {task_name: {metric: score, ...}, ...}.
    """
    try:
        return _run_via_library(port, tasks, limit, model_name)
    except Exception as e:
        print(f"Library invocation failed ({e}), falling back to CLI...")
        return _run_via_cli(port, tasks, limit, model_name)


def _run_via_library(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
) -> dict:
    """Use lm_eval.simple_evaluate() directly."""
    import lm_eval

    model_args = (
        f"model={model_name},"
        f"base_url=http://localhost:{port}/v1/completions,"
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
) -> dict:
    """Fall back to lm_eval CLI and parse JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            "--model_args", (
                f"model={model_name},"
                f"base_url=http://localhost:{port}/v1/completions,"
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
