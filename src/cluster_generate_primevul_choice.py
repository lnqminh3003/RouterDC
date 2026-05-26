import os
import sys
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from openTSNE import TSNE
import torch
from train_router_mdeberta import RouterDataset, RouterModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, DebertaV2Model
from sklearn.cluster import KMeans
import numpy as np
import random
import json

# ── cell 0: extract embeddings ────────────────────────────────────────────────
dataset_paths = ["../datasets/split2_primevul_choice/primevul_choice_train.json"]
data_types    = ["probability"]   # multiple choice → Eq. 2 scores

number_per_dataset = 12608

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

# ── cell 1: t-SNE ─────────────────────────────────────────────────────────────
np_hidden_states = all_hidden_states.cpu().numpy()
tsne_result      = TSNE(n_components=5, n_jobs=12).fit(np_hidden_states)
print(f"t-SNE result shape: {tsne_result.shape}")

# ── cell 2: k-means → save ────────────────────────────────────────────────────
n_clusters = 5   # paper uses N=5
seed = 42
random.seed(seed)
np.random.seed(seed)

base_output_path = "../datasets/split2_primevul_cluster_choice"
os.makedirs(base_output_path, exist_ok=True)

x      = tsne_result
kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
kmeans.fit(x)
kmeans_labels = kmeans.labels_.tolist()

# only one dataset so labels_split[0] = all labels
labels_split = [kmeans_labels[i * number_per_dataset: (i + 1) * number_per_dataset] for i in range(len(dataset_paths))]

for i, data_path in enumerate(dataset_paths):
    cluster_ids = labels_split[i]

    with open(data_path, 'r') as f:
        sample_list = json.load(f)

    new_sample_list = []
    for j, sample in enumerate(sample_list):   # no cutoff — use all samples
        new_sample = sample
        new_sample['cluster_id'] = cluster_ids[j]
        new_sample_list.append(new_sample)

    out_name = data_path.split('/')[-1].replace('.json', '_python_choice.json')
    out_path = os.path.join(base_output_path, out_name)
    with open(out_path, 'w') as f:
        json.dump(new_sample_list, f)

    print(f"Saved {len(new_sample_list)} samples to {out_path}")
    print(f"Cluster distribution: { {c: cluster_ids.count(c) for c in sorted(set(cluster_ids))} }")
