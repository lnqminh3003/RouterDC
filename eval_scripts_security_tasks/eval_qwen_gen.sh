model_path="Qwen/Qwen2.5-Coder-14B-Instruct"
model_name="Qwen2.5-Coder-14B-Instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True,gpu_memory_utilization=0.85" \
    --tasks primevul_gen \
    --batch_size auto \
    --output_path "./output/primevul_gen/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
