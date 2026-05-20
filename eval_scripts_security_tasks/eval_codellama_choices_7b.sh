model_path="codellama/CodeLlama-7b-Instruct-hf"
model_name="CodeLlama-7b-Instruct"

# primevul
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192" \
    --tasks primevul_choice_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_choice_1to2/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval