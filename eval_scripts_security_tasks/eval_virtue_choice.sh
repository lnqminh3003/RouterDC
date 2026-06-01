model_path="Virtue-AI-HUB/VulnLLM-R-7B"
model_name="VulnLLM-R-7B"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True,gpu_memory_utilization=0.55,tensor_parallel_size=2,dtype=bfloat16,enforce_eager=True,max_num_batched_tokens=8192" \
    --tasks primevul_choice \
    --batch_size 1 \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
