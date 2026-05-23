model_path="Qwen/Qwen2.5-Coder-14B-Instruct"
model_name="Qwen2.5-Coder-14B-Instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096,trust_remote_code=True,gpu_memory_utilization=0.90,tensor_parallel_size=2" \
    --tasks primevul_choice_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_choice_1to2/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval