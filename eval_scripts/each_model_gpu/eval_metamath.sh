model_path="meta-math/MetaMath-Mistral-7B"
model_name="MetaMath-Mistral-7B"

# gsm8k train
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks gsm8k_train \
    --batch_size 32 \
    --output_path "./output/gsm8k_train-t0.2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000

# gsm8k test
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks gsm8k-repeat10 \
    --batch_size 32 \
    --output_path "./output/gsm8k_test-t0.2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2"

# mmlu
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks mmlu \
    --batch_size auto \
    --output_path "./output/mmlu_5shot/${model_name}" \
    --log_samples \
    --num_fewshot 5

# arc
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks arc_challenge \
    --batch_size 32 \
    --output_path "./output/arc_challenge/${model_name}" \
    --log_samples

# cmmlu
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks cmmlu \
    --batch_size auto \
    --output_path "./output/cmmlu/${model_name}" \
    --log_samples \
    --num_fewshot 5

# ceval
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks ceval-valid \
    --batch_size auto \
    --output_path "./output/ceval-validation/${model_name}" \
    --log_samples \
    --num_fewshot 5

# humaneval
save_path="./output/humaneval/${model_name}/"
accelerate launch ./evaluation_LLM_tools/bigcode-evaluation-harness/main.py \
  --model $model_path \
  --max_length_generation 512 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json"

# mbpp
save_path="./output/mbpp/${model_name}/"
accelerate launch ./evaluation_LLM_tools/bigcode-evaluation-harness/main.py \
  --model $model_path \
  --max_length_generation 2048 \
  --tasks mbpp \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json"
