model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
model_name="DeepSeek-Coder-V2-Lite-Instruct"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096,trust_remote_code=True" \
    --tasks primevul_gen_1to2 \
    --batch_size auto \
    --output_path "./output/primevul_gen_1to2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
