model_path="Virtue-AI-HUB/VulnLLM-R-7B"
model_name="VulnLLM-R-7B"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True,gpu_memory_utilization=0.85" \
    --tasks primevul_gen_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_gen_1to2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
