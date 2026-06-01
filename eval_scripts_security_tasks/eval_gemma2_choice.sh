model_path="google/gemma-2-9b-it"
model_name="Gemma-2-9B-IT"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=6000,trust_remote_code=True,tensor_parallel_size=2,gpu_memory_utilization=0.50,disable_custom_all_reduce=True,enforce_eager=True,max_num_batched_tokens=6000" \
    --tasks primevul_choice \
    --batch_size 1 \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval