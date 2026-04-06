# COMMAND

llmbench run \
--models \
"bartowski/google_gemma-4-26B-A4B-it-GGUF:google_gemma-4-26B-A4B-it-Q4_K_M.gguf" \
"HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q3_K_M.gguf" \
--gpu gpu0 \
--context-length 8192 \
--cache-type-k q4_0 \
--cache-type-v q4_0 \
--tasks humaneval,gsm8k,minerva_math

## Completed models at q4_0

"unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-Q8_K_XL.gguf" \
"unsloth/Qwen3.5-27B-GGUF:Qwen3.5-27B-UD-Q4_K_XL.gguf" \
"bartowski/Qwen_Qwen3.5-35B-A3B-GGUF:Qwen_Qwen3.5-35B-A3B-Q3_K_S.gguf" \
"bartowski/Qwen_Qwen3.5-35B-A3B-GGUF:Qwen_Qwen3.5-35B-A3B-Q3_K_XL.gguf" \
"Jackrong/Qwopus3.5-9B-v3-GGUF:Qwen3.5-9B.Q8_0.gguf" \
"Jackrong/Qwopus3.5-27B-v3-GGUF:Qwopus3.5-27B-v3-Q5_K_M.gguf" \
"unsloth/gemma-4-31B-it-GGUF:gemma-4-31B-it-UD-Q4_K_XL.gguf" \
"HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf" \
"HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" \
"HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf" \
"HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" \
"Guervency/deepseek-coder-v2-lite-16b-gguf:DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf" \
"matteogeniaccio/GLM-Z1-32B-0414-GGUF-fixed:GLM-Z1-32B-0414-Q4_K_M.gguf" \
"MaziyarPanahi/Qwen3-8B-GGUF:Qwen3-8B.Q6_K.gguf" \
"Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:qwen2.5-coder-32b-instruct-q5_k_m.gguf" \
"unsloth/gemma-4-26B-A4B-it-GGUF:gemma-4-26B-A4B-it-UD-Q6_K.gguf" \
"unsloth/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q5_K_M.gguf" \
"unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf" \

# Completed but I want to run with both GPUs?

"unsloth/Qwen3.5-27B-GGUF:Qwen3.5-27B-Q5_K_M.gguf" \
