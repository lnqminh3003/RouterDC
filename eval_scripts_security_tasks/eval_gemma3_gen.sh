model_path="google/gemma-3-12b-it"
model_name="Gemma-3-12B-IT"

export TRITON_CACHE_DIR=/tmp/triton_cache_$$

lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=8192,trust_remote_code=True" \
    --tasks primevul_gen \
    --batch_size auto \
    --output_path "./output/primevul_gen/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --include_path ./custom_tasks_lm_eval
