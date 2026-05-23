model_path="Qwen/Qwen2.5-Coder-14B-Instruct"
model_name="Qwen2.5-Coder-14B-Instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096,trust_remote_code=True,gpu_memory_utilization=0.65,tensor_parallel_size=2,enforce_eager=True,max_num_batched_tokens=4096" \
    --tasks primevul_choice_1to2 \
    --batch_size 8 \
    --output_path "./output/primevul_choice_1to2/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval