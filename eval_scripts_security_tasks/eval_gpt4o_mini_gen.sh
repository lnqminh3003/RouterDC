model_name="gpt-4o-mini"

lm_eval --model openai-chat-completions \
    --model_args "model=gpt-4o-mini" \
    --tasks primevul_gen \
    --batch_size auto \
    --output_path "./output/primevul_gen/${model_name}" \
    --limit 3057:3058 \
    --apply_chat_template \
    --use_cache ./cache/primevul_gen_gpt4o_mini \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
