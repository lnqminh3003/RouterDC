model_path="google/codegemma-7b-it"
model_name="CodeGemma-7B-IT"

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192" \
    --tasks primevul_gen2 \
    --batch_size auto \
    --output_path "./output/primevul_gen2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
