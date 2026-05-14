model_path="meta-math/MetaMath-Mistral-7B"
model_name="MetaMath-Mistral-7B"

# gsm8k train
lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks gsm8k_train \
    --batch_size 1 \
    --output_path "./output/gsm8k_train-t0.2/${model_name}" \
    --log_samples \
    --limit 1000

# gsm8k test
lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks gsm8k-repeat10 \
    --batch_size 1 \
    --output_path "./output/gsm8k_test-t0.2/${model_name}" \
    --log_samples

# mmlu
lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \
    --batch_size 1 \
    --output_path "./output/mmlu_5shot/${model_name}" \
    --log_samples \
    --num_fewshot 5

# arc
lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks arc_challenge \
    --batch_size 1 \
    --output_path "./output/arc_challenge/${model_name}" \
    --log_samples

# cmmlu
lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks cmmlu \
    --batch_size 1 \
    --output_path "./output/cmmlu/${model_name}" \
    --log_samples \
    --num_fewshot 5

# humaneval (requires bigcode-evaluation-harness)
save_path="./output/humaneval/${model_name}/"
python "/Users/minhle/Documents/MSc research/Reproduce papers/RouterDC/evaluation_LLM_tools/bigcode-evaluation-harness/main.py" \
  --model $model_path \
  --max_length_generation 512 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 1 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json"
