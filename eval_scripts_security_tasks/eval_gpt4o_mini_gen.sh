model_name="gpt-4o-mini"

lm_eval --model openai-chat-completions \
    --model_args "model=gpt-4o-mini" \
    --tasks primevul_gen2 \
    --batch_size auto \
    --output_path "./output/primevul_gen2/${model_name}" \
    --apply_chat_template \
    --use_cache ./cache/primevul_gen2_gpt4o_mini \
    --log_samples \
    --include_path ./custom_tasks_lm_eval
