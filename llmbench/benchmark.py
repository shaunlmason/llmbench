import os
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# Allow HumanEval code execution
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


# Map GGUF filename prefixes to HuggingFace tokenizer repos
TOKENIZER_MAP = {
    "Qwen3.5-": "Qwen/Qwen3.5-4B-Base",
    "Qwen3-Coder-": "Qwen/Qwen3-Coder-0.6B",
    "Qwen3-": "Qwen/Qwen3-0.6B",
    "Qwen2.5-Coder-": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen2.5-": "Qwen/Qwen2.5-0.5B",
    "gemma-4-": "google/gemma-3-4b-it",
    "gemma-3-": "google/gemma-3-4b-it",
    "Mistral-": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3": "meta-llama/Llama-3.2-1B",
    "Phi-": "microsoft/Phi-3-mini-4k-instruct",
    "nvidia_Nemotron-": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Nemotron-": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "DeepSeek-Coder-V2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "GLM-Z1-": "THUDM/GLM-Z1-32B-0414",
    "GLM-4-": "THUDM/glm-4-9b",
}


def _repo_exists(repo_id: str) -> bool:
    """Check if a HuggingFace repo exists by hitting the API."""
    try:
        resp = requests.head(
            f"https://huggingface.co/api/models/{repo_id}",
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _resolve_tokenizer(repo_id: str, model_name: str, tokenizer: str | None) -> str:
    """Resolve a HuggingFace tokenizer repo.

    Priority:
    1. Explicit --tokenizer flag
    2. Strip -GGUF from the GGUF repo ID (if the resulting repo exists)
    3. Filename prefix map as last resort
    """
    if tokenizer:
        return tokenizer
    # Most GGUF repos have a non-GGUF sibling with the tokenizer
    if repo_id.endswith("-GGUF"):
        candidate = repo_id.removesuffix("-GGUF")
        if _repo_exists(candidate):
            return candidate
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
    "gsm8k": "exact_match",
    "minerva_math": "exact_match",
    "mmlu": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "truthfulqa_mc2": "acc",
    "winogrande": "acc",
}

LOGPROB_OUTPUT_TYPES = {
    "loglikelihood",
    "loglikelihood_rolling",
    "multiple_choice",
}

# Task groups and their subtask counts — used to divide --limit evenly
_GROUP_SUBTASK_COUNTS = {
    "minerva_math": 7,  # algebra, counting_and_prob, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
}


def _effective_limit(task: str, limit: int) -> int:
    """For group tasks, divide limit by subtask count so total stays near the requested limit."""
    subtask_count = _GROUP_SUBTASK_COUNTS.get(task)
    if subtask_count:
        return max(1, limit // subtask_count)
    return limit


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

    runnable_tasks, skipped_tasks = _split_tasks_by_api_capability(port, tasks, model_name)
    if skipped_tasks:
        skipped = ", ".join(f"{name} ({reason})" for name, reason in skipped_tasks.items())
        print(f"Skipping unsupported tasks: {skipped}")
    if not runnable_tasks:
        details = ", ".join(f"{name} ({reason})" for name, reason in skipped_tasks.items())
        raise RuntimeError(
            "No runnable tasks remain for the current llama-server API. "
            f"Requested tasks: {details}"
        )

    # Run each task individually so we can report progress
    all_scores = {}
    total_start = time.monotonic()
    for i, task in enumerate(runnable_tasks, 1):
        task_limit = _effective_limit(task, limit)
        limit_note = f", limit={task_limit}" if task_limit != limit else ""
        print(f"\n[{i}/{len(runnable_tasks)}] Running {task}{limit_note}...")
        task_start = time.monotonic()
        try:
            scores = _run_via_library(port, [task], task_limit, model_name, tokenizer_repo)
        except Exception as e:
            print(f"Library invocation failed ({e}), falling back to CLI...")
            scores = _run_via_cli(port, [task], task_limit, model_name, tokenizer_repo)
        elapsed = time.monotonic() - task_start
        all_scores.update(scores)
        # Print score for this task
        for task_name, metrics in scores.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    print(f"  {task_name}: {v:.4f} ({elapsed:.0f}s)")
                    break
    total_elapsed = round(time.monotonic() - total_start)
    all_scores["_elapsed_seconds"] = total_elapsed
    print(f"\nTotal benchmark time: {total_elapsed // 60}m {total_elapsed % 60}s")
    return all_scores


def _task_requires_prompt_logprobs(tasks: list[str]) -> dict[str, bool]:
    """Resolve whether each requested task depends on prompt logprobs."""
    from lm_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager()
    requirements = {}
    for requested_task in tasks:
        task_dict = get_task_dict([requested_task], task_manager)
        if not task_dict:
            raise RuntimeError(f"Could not resolve task '{requested_task}'.")

        requires_prompt_logprobs = False
        for task_name, task in task_dict.items():
            output_type = getattr(task, "output_type", None)
            if output_type is None:
                config = getattr(task, "config", None)
                if config is not None:
                    output_type = getattr(config, "output_type", None)
            if output_type is None:
                # Task groups (e.g. minerva_math) don't have output_type; skip them
                continue
            if output_type in LOGPROB_OUTPUT_TYPES:
                requires_prompt_logprobs = True
                break

        requirements[requested_task] = requires_prompt_logprobs
    return requirements


def _server_supports_prompt_logprobs(port: int, model_name: str) -> bool:
    """Check whether the OpenAI-compatible endpoint returns echoed prompt logprobs."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": " hello",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "echo": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Prompt logprob capability probe failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        return False

    choice = choices[0]
    text = choice.get("text")
    logprobs = choice.get("logprobs")
    if not isinstance(text, str) or not text.startswith(payload["prompt"]):
        return False
    if not isinstance(logprobs, dict):
        return False

    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        return False

    return len(token_logprobs) >= 2


def _split_tasks_by_api_capability(
    port: int,
    tasks: list[str],
    model_name: str,
) -> tuple[list[str], dict[str, str]]:
    """Filter out tasks that need prompt logprobs when the server cannot provide them."""
    task_requirements = _task_requires_prompt_logprobs(tasks)
    prompt_logprob_tasks = [
        task_name
        for task_name, requires_prompt_logprobs in task_requirements.items()
        if requires_prompt_logprobs
    ]
    if not prompt_logprob_tasks:
        return list(tasks), {}

    if _server_supports_prompt_logprobs(port, model_name):
        return list(tasks), {}

    skipped_tasks = {
        task_name: "server did not return echoed prompt logprobs on /v1/completions"
        for task_name in prompt_logprob_tasks
    }
    runnable_tasks = [
        task_name
        for task_name, requires_prompt_logprobs in task_requirements.items()
        if not requires_prompt_logprobs
    ]
    return runnable_tasks, skipped_tasks


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
        confirm_run_unsafe_code=True,
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
            "--confirm_run_unsafe_code",
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
