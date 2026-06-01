model_path="google/codegemma-7b-it"
model_name="CodeGemma-7B-IT"


lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=5000,trust_remote_code=True,tensor_parallel_size=2,gpu_memory_utilization=0.50,disable_custom_all_reduce=True,enforce_eager=True,dtype=bfloat16,max_num_batched_tokens=5000" \
    --tasks primevul_choice \
    --batch_size 1 \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval