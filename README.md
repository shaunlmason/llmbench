# llmbench

Fast local LLM benchmarking for llama-server. Evaluates GGUF models on general reasoning (HellaSwag) and coding (HumanEval), with results tracked and ranked over time.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Benchmark a model on GPU0
llmbench run \
  --models "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  --gpu gpu0 \
  --context-length 4096

# Benchmark multiple models
llmbench run \
  --models "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
           "TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
  --gpu both \
  --context-length 8192

# View rankings
llmbench results

# Raw JSON output
llmbench results --json
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | required | HuggingFace refs (`repo:file.gguf`) |
| `--gpu` | `gpu0` | `gpu0`, `gpu1`, or `both` |
| `--context-length` | `4096` | Context window size |
| `--tasks` | `hellaswag,humaneval` | Comma-separated lm-eval tasks |
| `--limit` | `200` | Samples per task (speed knob) |
| `--models-dir` | `~/models` | GGUF storage directory |
| `--port` | `8080` | llama-server port |
| `--no-restore` | off | Skip restarting original services |

## How it works

For each model, llmbench:
1. Downloads the GGUF from HuggingFace (if needed)
2. Stops the running llama-server service(s)
3. Starts llama-server with the model on the specified GPU(s)
4. Runs benchmarks via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
5. Saves results to `~/.llmbench/history.json`
6. Restores the original services

Results are ranked by composite score (average of primary metrics across tasks).
