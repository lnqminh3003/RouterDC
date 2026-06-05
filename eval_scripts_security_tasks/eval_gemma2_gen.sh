model_path="google/gemma-2-9b-it"
model_name="Gemma-2-9B-IT"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True" \
    --tasks primevul_gen2 \
    --batch_size auto \
    --output_path "./output/primevul_gen2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
