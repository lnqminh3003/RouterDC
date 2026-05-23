model_path="Virtue-AI-HUB/VulnLLM-R-7B"
model_name="VulnLLM-R-7B"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True,tensor_parallel_size=2,gpu_memory_utilization=0.90" \
    --tasks primevul_choice_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_choice_1to2/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
