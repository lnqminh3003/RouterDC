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
    PYTHONPATH=. ./venv/bin/python3 -u train_files/train_router_primevul_gen_8models.py --training_steps ${training_steps} --top_k ${top_k} --last_k ${last_k} --learning_rate ${learning_rate} --eval_steps 50 --tempreture ${tempreture} --similarity_function ${similarity_function} --sample_loss_weight ${sample_loss_weight} --cluster_loss_weight ${cluster_loss_weight} --seed ${seed} --batch_size 64 --training_samples_per_dataset 8405 \
    --data_paths "./datasets/primevul/balance_8models/gen_cluster/train_shuffle2.json" \
    --test_data_paths "./datasets/primevul/balance_8models/gen_cluster/test_shuffle2.json" \
    --test_data_type "multi_attempt" \
    --final_eval_data_paths "./datasets/primevul/balance_8models/gen_cluster/test_shuffle2.json" \
    --final_eval_data_type "multi_attempt" \
    --save_path "./logs/router_primevul_gen_8models_shuffle2/seed${seed}"
done
