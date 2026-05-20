model_path="codellama/CodeLlama-13b-Instruct-hf"
model_name="CodeLlama-13b-Instruct"

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks primevul_gen_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_gen_1to2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
