model_path="google/codegemma-7b-it"
model_name="CodeGemma-7B-IT"

# primevul
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192" \
    --tasks primevul_choice \
    --batch_size auto \
    --output_path "./output/primevul_choice/${model_name}" \
    --log_samples \
    --include_path ./custom_tasks_lm_eval