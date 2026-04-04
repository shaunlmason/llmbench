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
        if primary_key and primary_key in metrics:
            primary_scores.append(metrics[primary_key])
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

    # Build table rows sorted by composite score (descending)
    rows = []
    for entry in sorted(history, key=lambda e: e.get("composite_score", 0), reverse=True):
        model = entry.get("model", "unknown")
        # Shorten model name for display
        if ":" in model:
            model = model.split(":")[1]

        gpu = entry.get("gpu_config", "?")
        ctx = entry.get("context_length", "?")
        composite = entry.get("composite_score", 0)
        ts = entry.get("timestamp", "?")[:19]

        # Collect individual task scores
        task_scores = []
        for task, metrics in entry.get("scores", {}).items():
            primary_key = PRIMARY_METRICS.get(task)
            if primary_key and primary_key in metrics:
                task_scores.append(f"{task}: {metrics[primary_key]:.3f}")
            else:
                # Show first numeric metric
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and "stderr" not in k:
                        task_scores.append(f"{task}: {v:.3f}")
                        break

        rows.append([
            model,
            gpu,
            ctx,
            f"{composite:.4f}",
            " | ".join(task_scores),
            ts,
        ])

    headers = ["Model", "GPU", "Ctx", "Score", "Tasks", "Date"]
    print()
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()
