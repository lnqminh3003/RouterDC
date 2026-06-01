model_path="google/codegemma-7b-it"
model_name="CodeGemma-7B-IT"

# primevul
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096,trust_remote_code=True,gpu_memory_utilization=0.50,tensor_parallel_size=2,dtype=bfloat16,enforce_eager=True,max_num_batched_tokens=4096" \
    --tasks primevul_choice \
    --batch_size 1 \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval