export CUDA_VISIBLE_DEVICES="6"

top_k=3
last_k=3
training_steps=1000
learning_rate="5e-5"
tempreture=1
similarity_function="cos"
sample_loss_weight=0
cluster_loss_weight=1
seeds=(1)

for seed in "${seeds[@]}"; do
    ./venv/bin/python3 -u train_router_mdeberta.py --training_steps ${training_steps} --top_k ${top_k} --last_k ${last_k} --learning_rate ${learning_rate} --eval_steps 50 --tempreture ${tempreture} --similarity_function ${similarity_function} --sample_loss_weight ${sample_loss_weight} --cluster_loss_weight ${cluster_loss_weight} --final_eval --seed ${seed} --batch_size 64 \
    --data_paths "./datasets/split2_model7_cluster/gsm8k-train.json" "./datasets/split2_model7_cluster/humaneval_train.json"  "./datasets/split2_model7_cluster/arc_challenge_train.json" "./datasets/split2_model7_cluster/cmmlu_train.json" "./datasets/split2_model7_cluster/mmlu_train.json" \
    --test_data_paths "./datasets/split2_model7/gsm8k-test.json" "./datasets/split2_model7/humaneval_test.json" "./datasets/split2_model7/arc_challenge_test.json" "./datasets/split2_model7/cmmlu_test.json" "./datasets/split2_model7/mmlu_test.json" \
    --save_path "./logs/RouterDC/run1" \
    --test_data_type "multi_attempt" "multi_attempt" "probability" "probability" "probability"
done
