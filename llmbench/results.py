import json
import tempfile
from pathlib import Path

from tabulate import tabulate

from .config import HISTORY_DIR, HISTORY_FILE


# Primary metric per task for composite scoring
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


def save_result(entry: dict, history_file: Path = HISTORY_FILE):
    """Append a benchmark result to the history file."""
    history_file.parent.mkdir(parents=True, exist_ok=True)

    history = load_results(history_file)

    # Compute composite score
    primary_scores = []
    for task, metrics in entry.get("scores", {}).items():
        primary_key = PRIMARY_METRICS.get(task)
        if primary_key:
            # lm-eval may append suffixes (e.g. "pass@1,create_test")
            for key, value in metrics.items():
                if key.startswith(primary_key) and "stderr" not in key:
                    primary_scores.append(value)
                    break
    entry["composite_score"] = round(sum(primary_scores) / len(primary_scores), 4) if primary_scores else 0.0

    history.append(entry)

    # Atomic write
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=history_file.parent,
        suffix=".json",
        delete=False,
    )
    try:
        json.dump(history, tmp, indent=2)
        tmp.close()
        Path(tmp.name).replace(history_file)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise


def load_results(history_file: Path = HISTORY_FILE) -> list[dict]:
    """Load benchmark history from JSON file."""
    if not history_file.exists():
        return []
    with open(history_file) as f:
        return json.load(f)


def print_ranking_table(history_file: Path = HISTORY_FILE, as_json: bool = False):
    """Print a ranked summary of all benchmark results."""
    history = load_results(history_file)
    if not history:
        print("No benchmark results found.")
        return

    if as_json:
        print(json.dumps(history, indent=2))
        return

    # Aggregate scores: collapse subtasks (e.g. minerva_math_algebra) into parent group
    aggregated = []
    for entry in history:
        scores = entry.get("scores", {})
        collapsed = {}
        groups = {}
        for task, metrics in scores.items():
            # Check if this is a subtask (e.g. minerva_math_algebra -> minerva_math)
            parent = _find_parent_group(task)
            if parent:
                groups.setdefault(parent, [])
                score = _get_primary_score(task, metrics)
                if score is not None:
                    groups[parent].append(score)
            else:
                collapsed[task] = metrics
        # Average subtask scores into group score
        for group, sub_scores in groups.items():
            if sub_scores:
                avg = sum(sub_scores) / len(sub_scores)
                collapsed[group] = {"_avg": round(avg, 4)}
        aggregated.append({**entry, "_display_scores": collapsed})

    # Collect display task names
    all_tasks = []
    seen = set()
    for entry in aggregated:
        for task in entry["_display_scores"]:
            if task not in seen:
                all_tasks.append(task)
                seen.add(task)

    # Build table rows sorted by composite score (descending)
    rows = []
    for entry in sorted(aggregated, key=lambda e: e.get("composite_score", 0), reverse=True):
        model = entry.get("model", "unknown")
        if ":" in model:
            model = model.split(":")[1]
        # Trim .gguf suffix
        if model.endswith(".gguf"):
            model = model[:-5]

        gpu = entry.get("gpu_config", "?")
        ctx = entry.get("context_length", "?")
        lim = entry.get("limit", "-")
        composite = entry.get("composite_score", 0)

        elapsed = entry.get("elapsed_seconds")
        if elapsed:
            mins = elapsed // 60
            secs = elapsed % 60
            time_str = f"{mins}m{secs:02d}s"
            eff = composite / (elapsed / 60) if composite and elapsed else 0
            eff_str = f"{eff:.3f}"
        else:
            time_str = "-"
            eff_str = "-"

        server_args = entry.get("server_args", "")
        row = [model, gpu, ctx, lim, f"{composite:.4f}", time_str, eff_str]

        for task in all_tasks:
            metrics = entry["_display_scores"].get(task, {})
            score = _get_primary_score(task, metrics)
            row.append(f"{score:.3f}" if score is not None else "-")

        row.append(server_args)
        rows.append(row)

    # Short display names for column headers
    headers = ["Model", "GPU", "Ctx", "N", "Avg", "Time", "Eff"] + [_short_name(t) for t in all_tasks] + ["Args"]
    print()
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


# Known task groups — subtasks will be collapsed into the parent
_TASK_GROUPS = ["minerva_math"]


def _find_parent_group(task: str) -> str | None:
    """If task is a subtask of a known group, return the group name."""
    for group in _TASK_GROUPS:
        if task.startswith(group + "_"):
            return group
    return None


# Short display names for table headers
_SHORT_NAMES = {
    "humaneval": "human",
    "hellaswag": "hella",
    "minerva_math": "math",
    "gsm8k": "gsm8k",
    "mbpp": "mbpp",
    "mmlu": "mmlu",
    "arc_easy": "arc_e",
    "arc_challenge": "arc_c",
    "truthfulqa_mc2": "tqa",
    "winogrande": "wino",
}


def _short_name(task: str) -> str:
    return _SHORT_NAMES.get(task, task)


def _get_primary_score(task: str, metrics: dict) -> float | None:
    """Extract the primary score for a task from its metrics dict."""
    primary_key = PRIMARY_METRICS.get(task)
    if primary_key:
        for k, v in metrics.items():
            if k.startswith(primary_key) and "stderr" not in k:
                return v
    # Fall back to first numeric non-stderr metric
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return v
    return None
