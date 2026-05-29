model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
model_name="DeepSeek-Coder-V2-Lite-Instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096,trust_remote_code=True,tensor_parallel_size=2,gpu_memory_utilization=0.65,enforce_eager=True,max_num_batched_tokens=4096" \
    --tasks primevul_choice \
    --batch_size 8 \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
