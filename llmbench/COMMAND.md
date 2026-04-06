llmbench run \
 --models \
 "bartowski/google_gemma-4-26B-A4B-it-GGUF:google_gemma-4-26B-A4B-it-Q4_K_M.gguf" \
"bartowski/Qwen_Qwen3.5-35B-A3B-GGUF:Qwen_Qwen3.5-35B-A3B-Q3_K_S.gguf" \
 "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF:Qwen_Qwen3.5-35B-A3B-Q3_K_XL.gguf" \
 "bartowski/starcoder2-15b-instruct-v0.1-GGUF:starcoder2-15b-instruct-v0.1-Q8_0.gguf" \
 "Guervency/deepseek-coder-v2-lite-16b-gguf:DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf" \
 "Jackrong/Qwopus3.5-27B-v3-GGUF:Qwopus3.5-27B-v3-Q4_K_M.gguf" \
 "MaziyarPanahi/Qwen3-8B-GGUF:Qwen3-8B.Q6_K.gguf" \
 "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:qwen2.5-coder-32b-instruct-q5_k_m.gguf" \
 "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:Devstral-Small-2-24B-Instruct-2512-UD-Q5_K_XL.gguf" \
 "unsloth/gemma-4-26B-A4B-it-GGUF:gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" \
 "unsloth/gpt-oss-20b-GGUF:gpt-oss-20b-UD-Q8_K_XL.gguf" \
 "unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Nemotron-3-Nano-30B-A3B-UD-Q3_K_XL.gguf" \
 "unsloth/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q5_K_M.gguf" \
 "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf" \
 --gpu gpu0 \
 --context-length 8192 \
 --cache-type-k q4_0 \
 --cache-type-v q4_0 \
 --tasks humaneval,mbpp,gsm8k \
 --chat --no-think \
 --system "Output only the requested answer with no explanation, no commentary, and no markdown formatting. For code: emit only raw Python code without code fences or prose. For math: end your response with the final numeric answer wrapped in \\boxed{} on its own line." \
 --limit 50

## Potential

"bartowski/starcoder2-15b-instruct-v0.1-GGUF:starcoder2-15b-instruct-v0.1-Q8_0.gguf" \
 "bigatuna/Qwen3.5-27b-Sushi-Coder-RL-GGUF:qwen35-codeforces-27b-rl-step25-Q4_K_M.gguf" \
 "Guervency/deepseek-coder-v2-lite-16b-gguf:DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf" \
 "HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q3_K_M.gguf" \
 "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf" \
 "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" \
 "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf" \
 "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" \
 "Jackrong/Qwopus3.5-27B-v3-GGUF:Qwopus3.5-27B-v3-Q4_K_M.gguf" \
 "Jackrong/Qwopus3.5-27B-v3-GGUF:Qwopus3.5-27B-v3-Q5_K_M.gguf" \
 "Jackrong/Qwopus3.5-27B-v3-GGUF:Qwopus3.5-27B-v3-Q6_K.gguf" \
 "Jackrong/Qwopus3.5-9B-v3-GGUF:Qwen3.5-9B.Q8_0.gguf" \
 "juanml82/Qwen3.5-27B-heretic-gguf:Qwen3.5-27B-heretic-Q5_K_M.gguf" \
 "matteogeniaccio/GLM-Z1-32B-0414-GGUF-fixed:GLM-Z1-32B-0414-Q4_K_M.gguf" \
 "MaziyarPanahi/Qwen3-8B-GGUF:Qwen3-8B.Q6_K.gguf" \
 "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:qwen2.5-coder-32b-instruct-q5_k_m.gguf" \
 "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:Devstral-Small-2-24B-Instruct-2512-UD-Q5_K_XL.gguf" \
 "unsloth/gemma-4-26B-A4B-it-GGUF:gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" \
 "unsloth/gemma-4-26B-A4B-it-GGUF:gemma-4-26B-A4B-it-UD-Q6_K.gguf" \
 "unsloth/gemma-4-31B-it-GGUF:gemma-4-31B-it-UD-Q4_K_XL.gguf" \
 "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-Q8_K_XL.gguf" \
 "unsloth/gpt-oss-20b-GGUF:gpt-oss-20b-UD-Q8_K_XL.gguf" \
 "unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Nemotron-3-Nano-30B-A3B-UD-Q3_K_XL.gguf" \
 "unsloth/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q5_K_M.gguf" \
 "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf" \
 "unsloth/Qwen3.5-27B-GGUF:Qwen3.5-27B-Q5_K_M.gguf" \
 "unsloth/Qwen3.5-27B-GGUF:Qwen3.5-27B-UD-Q4_K_XL.gguf" \
