model_path="bigcode/starcoder2-15b-instruct-v0.1"
model_name="starcoder2-15b-instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True,gpu_memory_utilization=0.85" \
    --tasks primevul_choice_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_choice_1to2/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
