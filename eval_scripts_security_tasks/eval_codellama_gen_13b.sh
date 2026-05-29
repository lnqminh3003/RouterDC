model_path="codellama/CodeLlama-13b-Instruct-hf"
model_name="CodeLlama-13b-Instruct"

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,gpu_memory_utilization=0.95" \
    --tasks primevul_gen \
    --batch_size auto \
    --output_path "./output/primevul_gen/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
