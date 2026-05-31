model_name="gpt-4o-mini"

lm_eval --model openai-chat-completions \
    --model_args "model=gpt-4o-mini" \
    --tasks primevul_gen \
    --batch_size auto \
    --output_path "./output/primevul_gen/${model_name}" \
    --limit 10 \
    --apply_chat_template \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
