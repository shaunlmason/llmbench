import os
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# Allow HumanEval code execution
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


# Reasoning marker formats that llama-server doesn't strip on its own.
# These leak into chat completion responses and break code-extraction regexes
# in lm-eval (humaneval_instruct/mbpp_instruct), tanking scores even when the
# model answers correctly. The instruct task filters scan for ``` markdown
# fences; if the response starts with channel markers, the fence detection
# misfires and either returns empty or includes garbage.
_REASONING_PATTERNS = [
    # gemma-4 channel format: <|channel>thought\n<channel|>actual content
    # Note the asymmetric brackets: open is <|channel> close is <channel|>.
    re.compile(r"<\|channel\|?>\s*thought.*?<channel\|?>\s*", re.DOTALL),
    re.compile(r"<channel\|?>\s*thought.*?<\|channel\|?>\s*", re.DOTALL),

    # DeepSeek-R1 / Qwopus / Qwen3-with-reasoning / QwQ
    re.compile(r"<think>.*?</think>\s*", re.DOTALL),

    # Generic reasoning tag formats some fine-tunes use
    re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL),
    re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL),

    # OpenAI harmony format (gpt-oss): strip the analysis and commentary
    # channels entirely; keep only the final channel.
    # Format: <|start|>assistant<|channel|>analysis<|message|>...<|end|>
    re.compile(
        r"<\|start\|>[^<]*?<\|channel\|>analysis<\|message\|>.*?<\|end\|>\s*",
        re.DOTALL,
    ),
    re.compile(
        r"<\|start\|>[^<]*?<\|channel\|>commentary<\|message\|>.*?<\|end\|>\s*",
        re.DOTALL,
    ),
    # For the final channel, strip just the wrapper so the inner content is exposed.
    re.compile(
        r"<\|start\|>[^<]*?<\|channel\|>final<\|message\|>",
        re.DOTALL,
    ),
    # Stray harmony control tokens
    re.compile(r"<\|return\|>\s*", re.DOTALL),
    re.compile(r"<\|end\|>\s*", re.DOTALL),
]


def _strip_reasoning_markers(text):
    """Remove reasoning/thinking blocks from a model response string."""
    if not isinstance(text, str):
        return text
    for pattern in _REASONING_PATTERNS:
        text = pattern.sub("", text)
    return text.lstrip()


def _install_reasoning_stripper():
    """Patch lm-eval's LocalChatCompletion.parse_generations to strip reasoning
    markers from responses before they reach lm-eval's task filters.

    Without this, models that emit reasoning channels (gemma-4 <|channel>thought,
    Qwopus <think>, gpt-oss harmony, etc.) score 0 on chat-mode code benchmarks
    because the markers prefix the actual code and break extraction regexes
    (e.g. humaneval_instruct's filter scans for ``` fences and won't find them
    if the response starts with a channel marker).
    """
    try:
        from lm_eval.models.openai_completions import LocalChatCompletion
    except ImportError:
        print(
            "Warning: could not locate lm_eval.models.openai_completions.LocalChatCompletion. "
            "Reasoning marker stripping disabled — chat-mode code benchmarks may score 0 "
            "for reasoning models."
        )
        return

    if getattr(LocalChatCompletion, "_llmbench_reasoning_patched", False):
        return  # Already patched

    # parse_generations is a @staticmethod taking (outputs, **kwargs).
    # Accessing it via the class unwraps the descriptor, so we get the raw function.
    original = LocalChatCompletion.parse_generations

    def patched_parse_generations(outputs, **kwargs):
        results = original(outputs, **kwargs)
        if isinstance(results, list):
            return [_strip_reasoning_markers(r) for r in results]
        return _strip_reasoning_markers(results)

    # Re-wrap as staticmethod so Python doesn't try to bind self when called via instance.
    LocalChatCompletion.parse_generations = staticmethod(patched_parse_generations)
    LocalChatCompletion._llmbench_reasoning_patched = True
    print("Installed reasoning marker stripper for chat-mode benchmarks")


# Matches a fenced markdown code block, optional language tag, captures the body.
# Used to extract code from chat-mode model responses for humaneval/mbpp scoring.
_CODE_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+\-]*)\s*\n?(.*?)```",
    re.DOTALL,
)


def _extract_code_from_response(response: str) -> str:
    """Extract code from a chat-mode model response.

    Strategy: find all fenced code blocks (```lang\n...\n```), return the content
    of the LAST one (models often write explanation, then the code, then a
    summary that may itself contain a tiny snippet — we want the main solution).
    Falls back to the whole response if no fence is found.
    """
    if not isinstance(response, str):
        return response
    matches = _CODE_FENCE_RE.findall(response)
    if matches:
        # Strip trailing newlines but keep internal indentation
        return matches[-1].rstrip("\n")
    return response


def _patched_mbpp_build_predictions(resps, docs):
    """Replacement for mbpp.utils.build_predictions in chat mode.

    Extracts the LAST fenced code block from each response. Falls back to
    the whole response if no fence is found.
    """
    return [[_extract_code_from_response(r) for r in resp] for resp in resps]


def _patched_humaneval_build_predictions_instruct(resps, docs):
    """Replacement for humaneval.utils.build_predictions_instruct in chat mode.

    Extracts the LAST fenced code block from each response and prepends the
    original prompt's function signature. This works whether the model wrote:
      (a) a complete function — the prompt's signature gets redefined (Python
          takes the second `def`), or
      (b) just the body — the indented body flows into the prompt's function
          definition above.
    """
    return [
        [
            doc["prompt"] + _extract_code_from_response(r)
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


# Map of (function_name, file_path_substring) -> replacement function.
# Used by _patch_task_filter_chains to identify captured filter_fn references
# inside FilterEnsemble.filters partials and substitute our patched versions.
# Only humaneval_instruct and mbpp_instruct are targeted — both are leaf
# tasks whose filter_list captures one filter_fn each.
_FUNCTION_REPLACEMENTS = [
    ("build_predictions_instruct", "humaneval/utils", _patched_humaneval_build_predictions_instruct),
    ("build_predictions", "mbpp/utils", _patched_mbpp_build_predictions),
]


def _find_replacement_for(fn):
    """Return our patched replacement for a captured filter function, or None.

    Identifies functions by name + source file path, since lm-eval loads each
    task's utils.py via importlib.util.spec_from_file_location, which creates
    module instances NOT registered in sys.modules. The captured function's
    __code__.co_filename is the only reliable identifier.
    """
    if not callable(fn):
        return None
    name = getattr(fn, "__name__", None)
    code = getattr(fn, "__code__", None)
    co_filename = getattr(code, "co_filename", "") if code else ""
    if not name:
        return None
    for target_name, file_substr, replacement in _FUNCTION_REPLACEMENTS:
        if name == target_name and file_substr in co_filename:
            return replacement
    return None


def _patch_task_filter_chains(task_dict) -> int:
    """Walk pre-loaded Task objects and replace targeted filter functions
    with our patched versions. Mutates task objects in-place. Returns the
    number of replacements performed (so callers can verify the patch took).

    Why we mutate the tasks directly instead of patching at import: lm-eval's
    `!function` YAML constructor loads task utils.py via spec_from_file_location,
    so each task gets its own module instance NOT in sys.modules. The filter
    function reference is captured at YAML parse time inside a `functools.partial`
    that lives in `task._filters[i].filters[j].keywords['filter_fn']`. Replacing
    it directly in the loaded task objects is the only reliable way to intercept.

    Why this is needed: humaneval_instruct and mbpp_instruct were designed
    around `gen_prefix`, which prefills the start of the assistant response in
    COMPLETION mode. The OpenAI chat completions API has no way to prefill an
    assistant turn, so the gen_prefix is silently dropped — and the model emits
    its OWN ```python opening fence as part of a normal markdown reply. The
    original extractors then misbehave: mbpp returns "" because of double
    fences; humaneval returns just doc["prompt"] because it cuts at the first
    ``` (which is the opening fence, not the closing one).
    """
    import functools

    replacements = 0

    def _walk(task_dict_or_obj):
        nonlocal replacements
        if isinstance(task_dict_or_obj, dict):
            for v in task_dict_or_obj.values():
                _walk(v)
            return

        task = task_dict_or_obj
        task_name = getattr(getattr(task, "config", None), "task", None) or type(task).__name__

        # Patch task._filters (the live filter chain used at evaluation time)
        filters = getattr(task, "_filters", None)
        if filters:
            for i, ensemble in enumerate(filters):
                sub_filters = getattr(ensemble, "filters", None)
                if not sub_filters:
                    continue
                for j, partial_obj in enumerate(sub_filters):
                    if not isinstance(partial_obj, functools.partial):
                        continue
                    kw = dict(getattr(partial_obj, "keywords", None) or {})
                    fn = kw.get("filter_fn")
                    replacement = _find_replacement_for(fn)
                    if replacement is None:
                        continue
                    kw["filter_fn"] = replacement
                    new_partial = functools.partial(
                        partial_obj.func,
                        *(partial_obj.args or ()),
                        **kw,
                    )
                    sub_filters[j] = new_partial
                    replacements += 1
                    print(
                        f"  [filter patch] {task_name}._filters[{i}].filters[{j}]: "
                        f"{fn.__name__} -> llmbench replacement"
                    )

        # Also patch task.config.filter_list for consistency (some lm-eval
        # codepaths re-read this dict — keep both views aligned).
        cfg = getattr(task, "config", None)
        if cfg is not None:
            filter_list = getattr(cfg, "filter_list", None)
            if filter_list:
                for fc in filter_list:
                    for fn_cfg in fc.get("filter", []):
                        if not isinstance(fn_cfg, dict):
                            continue
                        fn_ref = fn_cfg.get("filter_fn")
                        replacement = _find_replacement_for(fn_ref)
                        if replacement is not None:
                            fn_cfg["filter_fn"] = replacement

    _walk(task_dict)
    return replacements


def _preload_and_patch_chat_tasks(tasks: list[str]):
    """Pre-load tasks via TaskManager + get_task_dict, then walk each task's
    filter chain and replace captured filter functions with our chat-aware
    versions. Returns (task_objects_list, task_manager) to pass to
    simple_evaluate.

    Returns (None, None) if lm_eval is not importable or no tasks needed
    patching — caller should fall back to passing string task names.
    """
    try:
        from lm_eval.tasks import TaskManager, get_task_dict
    except ImportError:
        print("Warning: could not import lm_eval.tasks — chat-mode code tasks may score 0.")
        return None, None

    task_manager = TaskManager()
    task_dict = get_task_dict(tasks, task_manager)

    replacements = _patch_task_filter_chains(task_dict)
    if replacements == 0:
        # Nothing to patch — let simple_evaluate handle the tasks normally
        return None, None

    # Flatten any nested groups into a list of task objects for simple_evaluate
    task_objects = []

    def _collect(node):
        if isinstance(node, dict):
            for v in node.values():
                _collect(v)
        else:
            task_objects.append(node)

    _collect(task_dict)
    print(f"Patched {replacements} filter function(s) for chat-mode code extraction")
    return task_objects, task_manager


# Map GGUF filename prefixes to HuggingFace tokenizer repos
TOKENIZER_MAP = {
    "Bonsai-": "Qwen/Qwen3-0.6B",
    "Qwopus3.5-": "Qwen/Qwen3.5-4B-Base",
    "Qwen_Qwen3.5-": "Qwen/Qwen3.5-4B-Base",
    "Qwen3.5-": "Qwen/Qwen3.5-4B-Base",
    "Qwen3-Coder-": "Qwen/Qwen3-Coder-0.6B",
    "Qwen3-": "Qwen/Qwen3-0.6B",
    "Qwen2.5-Coder-": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen2.5-": "Qwen/Qwen2.5-0.5B",
    "google_gemma-4-": "unsloth/gemma-4-E4B-it",
    "gemma-4-": "unsloth/gemma-4-E4B-it",
    "google_gemma-3-": "google/gemma-3-4b-it",
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
    repo_upper = repo_id.upper()
    if repo_upper.endswith("-GGUF"):
        candidate = repo_id[:-5]  # strip last 5 chars (-GGUF/-gguf/etc)
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
    "gsm8k": "exact_match,flexible-extract",
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

# In chat mode, completion-style code tasks fail because their extraction
# regexes assume the model continues the function body. The _instruct variants
# expect a fenced markdown code block, which is what chat-tuned models naturally
# emit. Auto-swap when chat mode is on.
_CHAT_TASK_SWAPS = {
    "humaneval": "humaneval_instruct",
    "mbpp": "mbpp_instruct",
}


def _effective_limit(task: str, limit: int) -> int:
    """For group tasks, divide limit by subtask count so total stays near the requested limit."""
    subtask_count = _GROUP_SUBTASK_COUNTS.get(task)
    if subtask_count:
        return max(1, limit // subtask_count)
    return limit


def _swap_chat_tasks(tasks: list[str], chat: bool) -> tuple[list[str], dict[str, str]]:
    """In chat mode, swap completion-style code tasks for their _instruct variants.

    Returns (swapped_tasks, restore_map) where restore_map[swapped] = original
    so callers can rename score keys back to the original task names for display.
    """
    if not chat:
        return list(tasks), {}
    swapped_tasks = []
    restore_map = {}
    for task in tasks:
        replacement = _CHAT_TASK_SWAPS.get(task)
        if replacement:
            swapped_tasks.append(replacement)
            restore_map[replacement] = task
            print(f"Chat mode: swapping {task} -> {replacement} (chat-aware variant)")
        else:
            swapped_tasks.append(task)
    return swapped_tasks, restore_map


def _restore_swapped_score_keys(scores: dict, restore_map: dict[str, str]) -> dict:
    """Rename swapped task names in a scores dict back to their originals."""
    if not restore_map:
        return scores
    return {restore_map.get(name, name): metrics for name, metrics in scores.items()}


def run_benchmark(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
    repo_id: str,
    tokenizer: str | None = None,
    chat: bool = False,
    system: str | None = None,
) -> dict:
    """Run lm-evaluation-harness against the local llama-server endpoint.

    Tries the Python API first, falls back to CLI subprocess.
    Returns {task_name: {metric: score, ...}, ...}.
    """
    tokenizer_repo = _resolve_tokenizer(repo_id, model_name, tokenizer)
    print(f"Using tokenizer: {tokenizer_repo}")
    if chat:
        print("Mode: chat completions (/v1/chat/completions)")
    if system:
        preview = system if len(system) <= 80 else system[:77] + "..."
        print(f"System instruction: {preview}")

    runnable_tasks, skipped_tasks = _split_tasks_by_api_capability(port, tasks, model_name, chat)
    if skipped_tasks:
        skipped = ", ".join(f"{name} ({reason})" for name, reason in skipped_tasks.items())
        print(f"Skipping unsupported tasks: {skipped}")
    if not runnable_tasks:
        details = ", ".join(f"{name} ({reason})" for name, reason in skipped_tasks.items())
        raise RuntimeError(
            "No runnable tasks remain for the current llama-server API. "
            f"Requested tasks: {details}"
        )

    # Auto-swap completion-style code tasks (humaneval, mbpp) for their _instruct
    # variants when running in chat mode. The instruct variants extract code from
    # markdown fences, which is what chat-tuned models naturally produce.
    runnable_tasks, restore_map = _swap_chat_tasks(runnable_tasks, chat)

    # Run each task individually so we can report progress
    all_scores = {}
    total_start = time.monotonic()
    for i, task in enumerate(runnable_tasks, 1):
        task_limit = _effective_limit(task, limit)
        limit_note = f", limit={task_limit}" if task_limit != limit else ""
        print(f"\n[{i}/{len(runnable_tasks)}] Running {task}{limit_note}...")
        task_start = time.monotonic()
        try:
            scores = _run_via_library(port, [task], task_limit, model_name, tokenizer_repo, chat, system)
        except Exception as e:
            print(f"Library invocation failed ({e}), falling back to CLI...")
            scores = _run_via_cli(port, [task], task_limit, model_name, tokenizer_repo, chat, system)
        # Rename swapped task names back to their originals so existing PRIMARY_METRICS,
        # SHORT_NAMES, and history entries remain consistent across runs.
        scores = _restore_swapped_score_keys(scores, restore_map)
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
    chat: bool = False,
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

    if chat:
        # Chat completions endpoint never exposes echoed prompt logprobs.
        skipped_tasks = {
            task_name: "chat mode does not support logprob-based tasks"
            for task_name in prompt_logprob_tasks
        }
        runnable_tasks = [
            task_name
            for task_name, requires_prompt_logprobs in task_requirements.items()
            if not requires_prompt_logprobs
        ]
        return runnable_tasks, skipped_tasks

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


def _model_type_and_url(port: int, chat: bool) -> tuple[str, str]:
    if chat:
        return "local-chat-completions", f"http://localhost:{port}/v1/chat/completions"
    return "local-completions", f"http://localhost:{port}/v1/completions"


def _run_via_library(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
    tokenizer_repo: str,
    chat: bool = False,
    system: str | None = None,
) -> dict:
    """Use lm_eval.simple_evaluate() directly."""
    import lm_eval

    tasks_arg: list = list(tasks)
    task_manager_arg = None
    if chat:
        _install_reasoning_stripper()
        # Pre-load tasks so we can walk their filter chains and substitute
        # chat-aware code extractors before evaluation runs.
        patched_tasks, patched_tm = _preload_and_patch_chat_tasks(tasks)
        if patched_tasks is not None:
            tasks_arg = patched_tasks
            task_manager_arg = patched_tm

    model_type, base_url = _model_type_and_url(port, chat)
    model_args = (
        f"model={model_name},"
        f"base_url={base_url},"
        f"tokenizer_backend=huggingface,"
        f"tokenizer={tokenizer_repo},"
        f"num_concurrent=4,"
        f"max_gen_toks=512,"
        f"timeout=600"
    )

    print(f"Running benchmark: tasks={tasks}, limit={limit}")
    results = lm_eval.simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=tasks_arg,
        task_manager=task_manager_arg,
        limit=limit,
        log_samples=False,
        confirm_run_unsafe_code=True,
        apply_chat_template=chat,
        system_instruction=system,
    )

    return _extract_scores(results["results"])


def _run_via_cli(
    port: int,
    tasks: list[str],
    limit: int,
    model_name: str,
    tokenizer_repo: str,
    chat: bool = False,
    system: str | None = None,
) -> dict:
    """Fall back to lm_eval CLI and parse JSON output.

    Note: chat-mode code extraction patches (humaneval_instruct/mbpp_instruct)
    only work in the library path because they walk pre-loaded Task objects.
    The CLI runs lm_eval in a subprocess where our in-process patches don't
    reach. If chat-mode code tasks land here, they will score incorrectly.
    """
    if chat:
        _install_reasoning_stripper()
        if any(t in tasks for t in ("humaneval_instruct", "mbpp_instruct", "humaneval", "mbpp")):
            print(
                "Warning: CLI fallback cannot patch chat-mode code-extraction filters; "
                "humaneval/mbpp scores from this path will be unreliable."
            )
    model_type, base_url = _model_type_and_url(port, chat)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", model_type,
            "--model_args", (
                f"model={model_name},"
                f"base_url={base_url},"
                f"tokenizer_backend=huggingface,"
                f"tokenizer={tokenizer_repo},"
                f"num_concurrent=4,"
                f"max_gen_toks=512,"
                f"timeout=600"
            ),
            "--tasks", ",".join(tasks),
            "--limit", str(limit),
            "--output_path", tmpdir,
            "--confirm_run_unsafe_code",
        ]
        if chat:
            cmd.append("--apply_chat_template")
        if system:
            cmd.extend(["--system_instruction", system])

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
