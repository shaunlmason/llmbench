"""One-shot diagnostic to reveal how lm-eval stores filter function references.

Run from the venv where lm_eval is installed:
    python diagnose_filters.py

Output reveals:
  1. How many distinct module instances exist for humaneval/mbpp utils
  2. Where the captured build_predictions function reference lives in the
     task object (so we know what to patch)
  3. Whether the captured reference matches the module attribute
"""
import os

# code_eval refuses to load without this — humaneval/mbpp utils.py both
# call code_eval at import time as a smoke test.
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import sys


def show(label, value):
    print(f"  {label}: {value}")


def main():
    print("=" * 70)
    print("STEP 1: Pre-load tasks to force YAML parsing")
    print("=" * 70)

    from lm_eval.tasks import TaskManager, get_task_dict

    tm = TaskManager()
    task_names = ["humaneval_instruct", "mbpp_instruct"]
    task_dict = get_task_dict(task_names, tm)
    print(f"Loaded task_dict keys: {list(task_dict.keys())}")
    print()

    print("=" * 70)
    print("STEP 2: Walk sys.modules for humaneval/mbpp utils instances")
    print("=" * 70)
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        f = getattr(mod, "__file__", None) or ""
        if "tasks/humaneval/utils" in f or "tasks/mbpp/utils" in f:
            print(f"\nModule registered as: {name!r}")
            show("__file__", f)
            show("module id", id(mod))
            for attr in ("build_predictions", "build_predictions_instruct", "extract_code_blocks"):
                fn = getattr(mod, attr, None)
                if fn is not None:
                    show(f"{attr} id", f"{id(fn)} (qualname={fn.__qualname__})")
    print()

    print("=" * 70)
    print("STEP 3: Inspect each task object's filter chain")
    print("=" * 70)
    for task_name, task in task_dict.items():
        print(f"\n--- {task_name} ---")
        show("task type", type(task).__name__)
        show("task module", type(task).__module__)
        # Look for filter list attributes
        for attr_name in ("_filters", "filter_list", "_config", "config"):
            attr = getattr(task, attr_name, None)
            if attr is not None:
                show(f"has attr {attr_name}", type(attr).__name__)
        # Walk _filters specifically
        filters = getattr(task, "_filters", None)
        if filters:
            for i, f in enumerate(filters):
                print(f"  filter[{i}]: type={type(f).__name__}, repr={f!r}")
                # Filters are usually FilterEnsemble or similar — walk their internals
                for sub_attr in ("filters", "_filters", "filter_fn", "filter"):
                    sub = getattr(f, sub_attr, None)
                    if sub is not None:
                        print(f"    .{sub_attr}: {sub!r}")
                        if isinstance(sub, list):
                            for j, s in enumerate(sub):
                                print(f"      [{j}]: type={type(s).__name__}")
                                for ss_attr in ("filter_fn", "fn", "function", "_filter_fn"):
                                    ss = getattr(s, ss_attr, None)
                                    if ss is not None:
                                        if callable(ss):
                                            print(f"        .{ss_attr}: id={id(ss)} qualname={getattr(ss, '__qualname__', '?')}")
                                        else:
                                            print(f"        .{ss_attr}: {ss!r}")

        # Also check task.config for filter references
        cfg = getattr(task, "_config", None) or getattr(task, "config", None)
        if cfg is not None:
            cfg_filter_list = getattr(cfg, "filter_list", None) or (
                cfg.get("filter_list") if isinstance(cfg, dict) else None
            )
            if cfg_filter_list:
                print(f"  config.filter_list: {cfg_filter_list!r}")

    print()
    print("=" * 70)
    print("STEP 4: Compare module-level vs captured function identities")
    print("=" * 70)
    # Find any humaneval/mbpp utils module and grab build_predictions / build_predictions_instruct
    module_fns = {}
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        f = getattr(mod, "__file__", None) or ""
        if "tasks/humaneval/utils" in f:
            for attr in ("build_predictions", "build_predictions_instruct"):
                fn = getattr(mod, attr, None)
                if fn is not None:
                    module_fns[f"humaneval.{attr} (module={name})"] = id(fn)
        if "tasks/mbpp/utils" in f:
            for attr in ("build_predictions", "extract_code_blocks"):
                fn = getattr(mod, attr, None)
                if fn is not None:
                    module_fns[f"mbpp.{attr} (module={name})"] = id(fn)
    for k, v in module_fns.items():
        print(f"  {k}: id={v}")


if __name__ == "__main__":
    main()
