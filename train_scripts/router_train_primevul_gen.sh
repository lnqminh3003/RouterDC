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
    ./venv/bin/python3 -u train_router_primevul_gen.py --training_steps ${training_steps} --top_k ${top_k} --last_k ${last_k} --learning_rate ${learning_rate} --eval_steps 50 --tempreture ${tempreture} --similarity_function ${similarity_function} --sample_loss_weight ${sample_loss_weight} --cluster_loss_weight ${cluster_loss_weight} --seed ${seed} --batch_size 64 --training_samples_per_dataset 12608 \
    --data_paths "./datasets/split2_primevul_cluster_gen/primevul_gen_train.json" \
    --test_data_paths "./datasets/split2_primevul_gen/primevul_gen_test.json" \
    --test_data_type "multi_attempt" \
    --save_path "./logs/router_primevul_gen/seed${seed}"
done
