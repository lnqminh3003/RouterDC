import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import json
from openTSNE import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, DebertaV2Model

from train_router_mdeberta import RouterDataset, RouterModule

# ── config ────────────────────────────────────────────────────────────────────
dataset_paths      = [os.path.join(ROOT_DIR, "datasets/split2_primevul_gen/primevul_gen_train.json")]
data_types         = ["multi_attempt"]
number_per_dataset = 12608
n_clusters         = 5       # paper uses N=5
seed               = 42
base_output_path   = os.path.join(ROOT_DIR, "datasets/split2_primevul_cluster_gen")

random.seed(seed)
np.random.seed(seed)

# ── step 1: extract embeddings ────────────────────────────────────────────────
tokenizer     = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side='left', padding=True)
encoder_model = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base").to("cuda")

router_datasets = [RouterDataset(data_path, data_type=data_types[i], dataset_id=i, size=number_per_dataset) for i, data_path in enumerate(dataset_paths)]
for router_dataset in router_datasets:
    router_dataset.register_tokenizer(tokenizer)
router_dataset    = ConcatDataset(router_datasets)
router_dataloader = DataLoader(router_dataset, batch_size=64)

router_model = RouterModule(encoder_model, hidden_state_dim=768, node_size=len(router_datasets[0].router_node), similarity_function="cos").to("cuda")

all_hidden_states = []
with torch.no_grad():
    for i, batch in enumerate(router_dataloader):
        inputs, _, dataset_id, cluster_id = batch
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        _, hidden_states = router_model(**inputs)
        all_hidden_states.append(hidden_states.cpu())

all_hidden_states = torch.concat(all_hidden_states)
print(f"Embeddings shape: {all_hidden_states.shape}")

# ── step 2: t-SNE ─────────────────────────────────────────────────────────────
np_hidden_states = all_hidden_states.cpu().numpy()
tsne_result      = TSNE(n_components=5, n_jobs=12, negative_gradient_method='bh').fit(np_hidden_states)
print(f"t-SNE result shape: {tsne_result.shape}")

# ── step 3: k-means → save ────────────────────────────────────────────────────
x      = tsne_result
kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
kmeans.fit(x)
kmeans_labels = kmeans.labels_.tolist()

labels_split = [kmeans_labels[i * number_per_dataset: (i + 1) * number_per_dataset] for i in range(len(dataset_paths))]

os.makedirs(base_output_path, exist_ok=True)

for i, data_path in enumerate(dataset_paths):
    cluster_ids = labels_split[i]

    with open(data_path, 'r') as f:
        sample_list = json.load(f)

    new_sample_list = []
    for j, sample in enumerate(sample_list):
        new_sample = sample
        new_sample['cluster_id'] = cluster_ids[j]
        new_sample_list.append(new_sample)

    out_name = data_path.split('/')[-1].replace('.json', '_python.json')
    out_path = os.path.join(base_output_path, out_name)
    with open(out_path, 'w') as f:
        json.dump(new_sample_list, f)

    print(f"Saved {len(new_sample_list)} samples to {out_path}")
    print(f"Cluster distribution: { {c: cluster_ids.count(c) for c in sorted(set(cluster_ids))} }")
