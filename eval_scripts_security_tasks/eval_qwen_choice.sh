model_path="Qwen/Qwen2.5-Coder-14B-Instruct"
model_name="Qwen2.5-Coder-14B-Instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True,gpu_memory_utilization=0.55,tensor_parallel_size=2,dtype=bfloat16,enforce_eager=True,max_num_batched_tokens=2048" \
    --tasks primevul_choice \
    --batch_size 4 \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval