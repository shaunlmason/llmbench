# llmbench

Fast local LLM benchmarking for llama-server. Evaluates GGUF models on coding, math, and reasoning tasks, with results tracked and ranked over time.

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

# Run with specific tasks
llmbench run \
  --models "MaziyarPanahi/Qwen3-8B-GGUF:Qwen3-8B.Q6_K.gguf" \
  --gpu gpu0 \
  --tasks hellaswag,humaneval,gsm8k,minerva_math

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
| `--tokenizer` | auto-detected | HuggingFace tokenizer repo ID |
| `--no-restore` | off | Skip restarting original services |

## Supported tasks

| Task | Type | Metric | Logprobs? |
|------|------|--------|-----------|
| `hellaswag` | Commonsense reasoning | `acc_norm` | Yes |
| `humaneval` | Code generation (164 tasks) | `pass@1` | No |
| `mbpp` | Code generation (500 tasks) | `pass@1` | No |
| `gsm8k` | Grade school math | `exact_match` | No |
| `minerva_math` | Competition math | `exact_match` | No |
| `mmlu` | Multitask knowledge | `acc` | Yes |
| `arc_easy` | Science reasoning (easy) | `acc_norm` | Yes |
| `arc_challenge` | Science reasoning (hard) | `acc_norm` | Yes |
| `winogrande` | Coreference resolution | `acc` | Yes |

Tasks marked "Logprobs: Yes" require the server to return prompt logprobs in legacy format. If the server doesn't support this, those tasks are automatically skipped.

## How it works

For each model, llmbench:
1. Downloads the GGUF from HuggingFace (if needed)
2. Stops the running llama-server service(s)
3. Starts llama-server with the model on the specified GPU(s)
4. Runs benchmarks via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
5. Saves results to `~/.llmbench/history.json`
6. Restores the original services

Results are ranked by composite score (average of primary metrics across tasks).

## License

[MIT](LICENSE)
