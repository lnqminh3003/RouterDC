model_path="codellama/CodeLlama-7b-Instruct-hf"
model_name="CodeLlama-7b-Instruct"

# primevul
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192" \
    --tasks primevul \
    --batch_size auto \
    --output_path "./output/primevul/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval