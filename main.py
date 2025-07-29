import torch
from torchinfo import summary
import matplotlib.pyplot as plt
import time
import os
import glob
from pathlib import Path
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

from Data_Generator2 import generate_temporal_graph_dataset_new
from Model import TGCN_Autoencoder
from train_batch import train_model as train_model_batch, evaluate_model
from visulisation import plot_training_curves
from preparing_data import split_dataset, load_split_factors, create_windowed_splits, split_sum_into_3d
from parafac_fun import (parafac_decomposition_list_sparse, parafac_decomposition_sparse, run_sparse_parafac)
from Data_Generator3 import generate_stochastic_graph_with_communities_and_cycles, generate_random_temporal_graph
from model_summary import prepare_wrapped_model, show_model_summary
from biparatide import generate_bipartite_graph_with_cycles

# ==== 1. Preparing Dataset Parameters ====
nNodes = 5000          # Number of nodes in the graph
Time = 2000             # Number of time steps (snapshots)
Features = 1          # Feature dimensions per node
Rank = 3              # Rank of the feature matrix (used if applicable)

# Stochastic graph generation parameters
num_comm = 2          # Number of communities
num_cycle = 100         # Number of repeating cycles in graph pattern
num_samples = 1      # Number of graph sequences to generate

window_size = 100
num_window = Time // window_size

# Neural network architecture placeholder
layer_dims = [20]      # Example: input layer -> hidden layers -> output

base_path = "structural_anomalies5"
subfolders = ["train", "normal"]  # Generate data in both folders

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


gen_new_data = False
generate_parafac = False        # Whether to use sparse matrix representation
mode_training = False

# ==== 2. Generate Dataset in Both Folders ====
if gen_new_data:
    for folder in subfolders:
        # Prepare paths
        edge_path = f"{base_path}/{folder}/{num_samples}_temporal_graph_{Features}-{nNodes}-{Time}_.csv"
        feat_path = f"{base_path}/{folder}/{num_samples}_temporal_graph_{Features}-{nNodes}-{Time}_.pt"
        os.makedirs(os.path.dirname(edge_path), exist_ok=True)

        print(f"\n📁 Generating new dataset in: {folder}")
        print(f"→ Edges:    {edge_path}")
        print(f"→ Features: {feat_path}")

        start_time = time.time()

        # Call generation function
        dataset = generate_temporal_graph_dataset_new(
            nNodes=nNodes,
            Time=Time,
            Features=Features,
            num_cycle=num_cycle,
            num_comm=num_comm,
            num_samples=num_samples,
            high_prob=0.005,
            low_prob=0.0003,
            save_path_features=feat_path,
            save_path_edges=edge_path,
            visualize=False,
            node_i=5,
            node_j=1500,
            num_snapshots=10,
            feat_means=[4.1, 5.5, 1.2, 3.5],
            feat_ranges=[0.1, 0.4, 0.2, 0.3],
            device=device
        )

        elapsed = time.time() - start_time
        print(f"✅ Finished in {elapsed:.2f} seconds")

else:
    print("⚠️ Data generation skipped. Set `gen_new_data = True` to enable.")

print("✅ All Data Generation Completed")


#### ==== 2. Making the model
# Example of layer dimensions: input_dim=10, hidden layers, final output_dim=5
input_shape= (nNodes, window_size, Features)
# Create model
model = TGCN_Autoencoder(
    input_X= input_shape,
    layer_dims=layer_dims,
    nNodes=nNodes,
    Time=window_size,
    Rank=Rank,
    dropout=0.2,
    device=device
).to(device)

summary(model)


# Wrap and prepare
wrapped_model = prepare_wrapped_model(model, nNodes=nNodes, Time=window_size, Rank=Rank, device=device)

# Show model summary
show_model_summary(wrapped_model, nNodes=nNodes, Time=window_size, Features=Features, device=device)

#### ==== 3. preparing the anomalies data ==============================================================
# === FLAGS ===
generate_comm_cycle_anomalies = False
generate_bipartite_anomalies = True
generate_random_anomalies = False

# === BASE SETUP ===
base_path = Path("structural_anomalies5")
base_path.mkdir(exist_ok=True)
save_path_edges = []

# === 1. Community/Cycle-Based Anomalies ===
if generate_comm_cycle_anomalies:
    num_comm_list = [1, 1, 2, 4, 8, 10]
    num_cycle_list = [0, 500, 200, 500, 200, 200]  ## time // cycles ==> [0, 2, 5, 2, 5, 5] - [0, 4, 10, 4, 10, 10]

    for i, (num_comm, num_cycle) in enumerate(zip(num_comm_list, num_cycle_list), start=1):
        folder = base_path / f"anomaly{i}"
        folder.mkdir(parents=True, exist_ok=True)

        print(f"🔧 Generating Community/Cycle Anomaly {i} | comm: {num_comm}, cycle: {num_cycle}, time:{Time}")

        edge_path = generate_stochastic_graph_with_communities_and_cycles(
            nNodes=nNodes,
            Time=Time,
            Features=Features,
            num_comm=num_comm,
            num_cycle=num_cycle,
            high_prob=0.005,
            low_prob=0.0003,
            save_path_prefix=str(folder) + "/",  # trailing slash
            visualising=False,
            num_snapshots=10,
            node_i=2,
            node_j=5
        )

        save_path_edges.append(edge_path)

# === 2. Bipartite Graph Anomalies ===
if generate_bipartite_anomalies:
    p_edge_high_list = [0.003, 0.05, 0.1]
    p_edge_low_list = [0.0005, 0.0003, 0.001]
    num_cycles_list = [400, 200, 100]   ### [5, 10, 20]-[10, 20, 40]

    start_index = 7

    for i, (p_high, p_low, num_cycles) in enumerate(zip(p_edge_high_list, p_edge_low_list, num_cycles_list), start=start_index):
        folder = base_path / f"anomaly{i}"
        folder.mkdir(parents=True, exist_ok=True)

        print(f"🔧 Generating Bipartite Anomaly {i} | high: {p_high}, low: {p_low}, cycles: {num_cycles}, time:{Time}")

        edge_file = folder / f"bipartite_edges_cycle{num_cycles}_time{Time}.csv"
        # feature_file = folder / f"bipartite_features_cycle{num_cycles}_time{Time}.pt"

        generate_bipartite_graph_with_cycles(
            nU=nNodes//2,
            nV=nNodes//2,
            Time=Time,
            density_high=p_high,
            density_low=p_low,
            num_cycles=num_cycles,
            save_path_edges=edge_file,
            visualisation=False,
            vis_node_u=10,
            vis_node_v=20,
            vis_snapshots=5
        )

        save_path_edges.append(str(edge_file))

# === 3. Random Graph Anomalies ===
if generate_random_anomalies:
    edge_prob_list = [0.003, 0.03]

    # start_index = len(save_path_edges) + 1
    start_index = 10

    for i, edge_prob in enumerate(edge_prob_list, start=start_index):
        folder = base_path / f"anomaly{i}"
        folder.mkdir(parents=True, exist_ok=True)

        print(f"🔧 Generating Random Anomaly {i} | edge_prob: {edge_prob}, time:{Time}")

        edge_file = folder / f"random_edges_{edge_prob},time:{Time} .csv"
        feature_file = folder / f"random_features_{edge_prob},time:{Time}.pt"

        generate_random_temporal_graph(
            nNodes=nNodes,
            Time=Time,
            Features=Features,
            edge_prob=edge_prob,
            save_path_edges=str(edge_file),
            save_path_features=str(feature_file)
        )

        save_path_edges.append(str(edge_file))

# ✅ Summary
print("\n✅ All selected anomalies generated.")
print("Saved edge files:")
for path in save_path_edges:
    print(f"  - {path}")

####  Generating the PARAFAC agents --------------------------------------------------------------------
if generate_parafac:
    print("🔁 Starting PARAFAC decomposition for all .csv files...\n")

    # Walk through all subfolders of base_path
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                print(f"\n📄 Processing: {csv_path}")

                # # Create a subfolder to save factors
                # factors_folder = os.path.join(root, "factors")
                # os.makedirs(factors_folder, exist_ok=True)

                # Sliding window PARAFAC over time
                overlapping = str(0)
                step_size = window_size  # adjust overlap here
                for time_start in range(0, Time - window_size + 1, step_size):
                    time_end = time_start + window_size

                    # Save factors to a unique file in same folder
                    filename = root + f"/factors{overlapping}%-{time_start}-{time_end}-WS:{window_size}.pkl"

                    print(f"⏳ Decomposing: [{time_start}-{time_end}]")

                    start_time = time.time()

                    sorted_weights, sorted_factors = run_sparse_parafac(
                        csv_path, time_start, time_end, nNodes, filename,
                        rank=Rank, n_iter_max=100, tol=1e-5
                    )
                    print(f"\n✅ PARAFAC decompositions weights, {sorted_weights}.")
                    elapsed_time = time.time() - start_time
                    print(f"✅ Saved to: {filename} (⏱ {elapsed_time:.2f} sec)")


    print("\n✅ All PARAFAC decompositions completed.")


else:
    print("⚠️ generate_parafac=False — Skipping decomposition.")

input('---')
# === [1] Load Features and Factors === #

feature_path = "structural_anomalies5/train/1_temporal_graph_1-100-1000_.pt"
factors_dir = "structural_anomalies5/train/"
overlap_prefix = "factors0%-"

# Load windowed node features
train_features, val_features = create_windowed_splits(
    feature_path,
    window_size=window_size,
    train_ratio=0.7,
    val_ratio=0.3
)

# Load corresponding CP factor triplets [(A, B, C, weights), ...]
train_factors, val_factors = load_split_factors(
    factors_dir,
    overlap=overlap_prefix,
    train_ratio=0.7,
    val_ratio=0.3
)

# Debug shapes
print(f"[train] features: {train_features[0].shape}, factors: {len(train_factors)}, A.shape: {train_factors[0][0].shape}")
print(f"[val]   features: {val_features[0].shape}, factors: {len(val_factors)}, C.shape: {val_factors[0][2].shape}")


# === [2] Train the Model === #

if mode_training:
    print("🚀 Starting model training...")
    start_time = time.time()

    model, train_losses, val_losses, grad_history = train_model_batch(
        model=model,
        adj_list_train=train_factors,
        feat_list_train=train_features,
        adj_list_val=val_factors,
        feat_list_val=val_features,
        Rank=Rank,
        batch_size=len(train_features),
        epochs=1000,
        learning_rate=5e-4,
        patience=50,
        min_delta=1e-4,
        device=device,
        verbose=True,
        use_early_stopping=True
    )

    # Plot training curves
    plot_training_curves(train_losses, val_losses, log_axis=False)
    plt.show()

    elapsed_time = time.time() - start_time
    print(f"✅ Training completed in {elapsed_time:.2f} seconds")


### ===== 6. Evaluation


results_normal = 0.0
results_abnormal = 0.0

model = torch.load('best_model-50%overlap.pt', weights_only=False)

# A_true_val, B_true_val, C_true_val = adj_list_val[0]

print('🔵 Loading normal and abnormal samples (structural_anomalies3) ----------------------------------------')

# Helper function
def load_test_data(feature_path, factors_dir, overlap, has_features=True, label=""):
    if label:
        print(label)
    features, factors = None, None
    if has_features and feature_path is not None:
        _, features = create_windowed_splits(feature_path, window_size=window_size, train_ratio=0.0, val_ratio=1.0)
    _, factors = load_split_factors(factors_dir, overlap=overlap, train_ratio=0.0, val_ratio=1.0)
    return features, factors

# Normal sample
overlap = "factors0%-"
test_features_normal, test_factors_normal = load_test_data(
    "structural_anomalies4/normal/1_temporal_graph_1-100-1000_.pt",
    "structural_anomalies4/normal", overlap, True,
    "normal sample --------------------------------------------------------------------"
)

# Abnormal samples (all from structural_anomalies3 now)
_, test_factors_abnormal1 = load_test_data(
    "structural_anomalies4/anomaly1/_features_comm1_cycle0.pt",
    "structural_anomalies4/anomaly1/", overlap, True,
    "abnormal1 sample: 1 comm - 0 cycle ---------------------------------"
)

_, test_factors_abnormal2 = load_test_data(
    "structural_anomalies4/anomaly2/_features_comm1_cycle4.pt",
    "structural_anomalies4/anomaly2/", overlap, True,
    "abnormal2 sample: 1 comm 4 cycle -----------------------------------"
)

_, test_factors_abnormal3 = load_test_data(
    "structural_anomalies4/anomaly3/_features_comm2_cycle1.pt",
    "structural_anomalies4/anomaly3/", overlap, True,
    "abnormal3 sample: 2 comm 1 cycle ----------------------"
)

_, test_factors_abnormal4 = load_test_data(
    "structural_anomalies4/anomaly4/_features_comm4_cycle2.pt",
    "structural_anomalies4/anomaly4/", overlap, True,
    "abnormal4 sample: 4 comm 2 cycle ----------------------"
)

_, test_factors_abnormal5 = load_test_data(
    "structural_anomalies4/anomaly5/_features_comm10_cycle2.pt",
    "structural_anomalies4/anomaly5/", overlap, True,
    "abnormal5 sample: 10 comm 2 cycle ----------------------"
)

# Bipartite samples
_, test_factors_abnormal6 = load_test_data(
    "structural_anomalies4/anomaly6/bipartite_features_cycle4.pt",
    "structural_anomalies4/anomaly6/", overlap, True,
    "abnormal6 sample: bipartite 0 cycle ----------------------"
)

_, test_factors_abnormal7 = load_test_data(
    "structural_anomalies4/anomaly7/bipartite_features_cycle2.pt",
    "structural_anomalies4/anomaly7/", overlap, True,
    "abnormal7 sample: bipartite 2 cycle ----------------------"
)

_, test_factors_abnormal8 = load_test_data(
    "structural_anomalies4/anomaly8/bipartite_features_cycle1.pt",
    "structural_anomalies4/anomaly8/", overlap, True,
    "abnormal8 sample: bipartite 4 cycle ----------------------"
)

_, test_factors_abnormal9 = load_test_data(
    "structural_anomalies4/anomaly9/random_features_0.5.pt",
    "structural_anomalies4/anomaly9/", overlap, True,
    "abnormal9 sample: random edges 0.5 ----------------------"
)

_, test_factors_abnormal10 = load_test_data(
    "structural_anomalies4/anomaly10/random_features_0.1.pt",
    "structural_anomalies4/anomaly10/", overlap, True,
    "abnormal10 sample: random edges 0.1 ----------------------"
)


### ==================== PURE PARAFAC EVALUATION ==================== ###

print("📦 Extracting PARAFAC factors from NORMAL set...")
A_normal_list = [torch.tensor(f[0], dtype=torch.float32) for f in test_factors_normal]
B_normal_list = [torch.tensor(f[1], dtype=torch.float32) for f in test_factors_normal]
C_normal_list = [torch.tensor(f[2], dtype=torch.float32) for f in test_factors_normal]

# Average PARAFAC factors (reference)
A_avg = torch.stack(A_normal_list).mean(dim=0)
B_avg = torch.stack(B_normal_list).mean(dim=0)
C_avg = torch.stack(C_normal_list).mean(dim=0)

print(f"✅ Normal factor shapes: A={A_avg.shape}, B={B_avg.shape}, C={C_avg.shape}")

# Helper to extract abnormal PARAFAC factors
def extract_factors(name, factor_list):
    if not factor_list:
        print(f"⚠️ Skipping {name}: No data found.")
        return None
    A_list = [torch.tensor(f[0], dtype=torch.float32) for f in factor_list]
    B_list = [torch.tensor(f[1], dtype=torch.float32) for f in factor_list]
    C_list = [torch.tensor(f[2], dtype=torch.float32) for f in factor_list]
    print(f"{name} factor shapes: A={A_list[0].shape}, B={B_list[0].shape}, C={C_list[0].shape}")
    return (A_list, B_list, C_list)

# Extract abnormal PARAFAC factors
all_abnormals = []
for i in range(1, 11):
    factors = extract_factors(f"Abnormal{i}", eval(f"test_factors_abnormal{i}"))
    if factors:
        all_abnormals.append((i, factors))

# Compute MSE per sample
# criterion = nn.MSELoss(reduction='mean')

criterion = nn.L1Loss()

def compute_mse_per_sample(A_test, B_test, C_test, A_ref, B_ref, C_ref):
    mse_A = torch.stack([criterion(a, A_ref) for a in A_test])
    mse_B = torch.stack([criterion(b, B_ref) for b in B_test])
    mse_C = torch.stack([criterion(c, C_ref) for c in C_test])
    return mse_A + mse_B + mse_C

# Compute normal MSE
mse_normal = compute_mse_per_sample(A_normal_list, B_normal_list, C_normal_list, A_avg, B_avg, C_avg).detach().cpu().numpy()

# Evaluate all abnormal cases
print("\n📈 PARAFAC Evaluation Results:")
for idx, (A_ab, B_ab, C_ab) in all_abnormals:
    mse_ab = compute_mse_per_sample(A_ab, B_ab, C_ab, A_avg, B_avg, C_avg).detach().cpu().numpy()

    scores = np.concatenate([mse_normal, mse_ab])
    labels = np.array([0]*len(mse_normal) + [1]*len(mse_ab))

    auc = roc_auc_score(labels, scores)
    threshold = np.percentile(scores, 75)
    preds = (scores >= threshold).astype(int)

    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"🔸 Abnormal_{idx}: AUC={auc:.4f} | F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")


##### ============================


model = torch.load('best_model-saved.pt', map_location='cpu', weights_only=False)
# model.load_state_dict(state_dict)
model.eval()

# --- Prepare your data ---
# Normal data
test_features_normal, test_factors_normal = load_test_data(
    "structural_anomalies4/normal/1_temporal_graph_1-100-1000_.pt",
    "structural_anomalies4/normal", "factors0%-", True,
    "normal sample"
)

# Abnormal factors dict (change names to your variables)
abnormal_factors_dict = {
    1: test_factors_abnormal1,
    2: test_factors_abnormal2,
    3: test_factors_abnormal3,
    4: test_factors_abnormal4,
    5: test_factors_abnormal5,
    6: test_factors_abnormal6,
    7: test_factors_abnormal7,
    8: test_factors_abnormal8,
    9: test_factors_abnormal9,
    10: test_factors_abnormal10,
}

# --- Evaluate normal losses ---

normal_losses = []
for n in range(len(test_features_normal)):
    loss = evaluate_model(model, test_features_normal[n], test_factors_normal[n], Rank, device=device, verbose=False)
    normal_losses.append(loss)

print(f"Computed normal losses for {len(normal_losses)} samples.")

# --- Loop over abnormalities ---
for i in range(1, 11):
    abnormal_factors = abnormal_factors_dict.get(i, None)
    if not abnormal_factors or len(abnormal_factors) == 0:
        print(f"⚠️ Skipping Abnormal_{i} - no data available.")
        continue

    abnormal_losses = []
    for n in range(len(test_features_normal)):
        try:
            loss_ab = evaluate_model(model, test_features_normal[n], abnormal_factors[n], Rank, device=device, verbose=False)
            abnormal_losses.append(loss_ab)

        except Exception as e:
            print(f"❌ Error evaluating abnormal_{i} sample {n}: {e}")
            continue

    # Combine losses and labels
    scores = normal_losses + abnormal_losses
    labels = [0] * len(normal_losses) + [1] * len(abnormal_losses)

    # Compute metrics
    auc = roc_auc_score(labels, scores)
    threshold = np.mean(normal_losses) + 2 * np.std(normal_losses)
    preds = [1 if s >= threshold else 0 for s in scores]
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"\n📊 Evaluation for Abnormal_{i}:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(classification_report(labels, preds, target_names=["Normal", "Abnormal"]))

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist([normal_losses, abnormal_losses],
             bins=30,
             label=["Normal", f"Abnormal_{i}"],
             color=["skyblue", "salmon"],
             alpha=0.7,
             edgecolor='black')
    plt.axvline(np.mean(normal_losses), color='blue', linestyle='--', label='Mean Normal')
    plt.axvline(np.mean(abnormal_losses), color='red', linestyle='--', label='Mean Abnormal')
    plt.title(f"Loss Distribution - Abnormal_{i}")
    plt.xlabel("Loss Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for i in range(1, 11):
    abnormal_factors = abnormal_factors_dict.get(i, None)
    if not abnormal_factors or len(abnormal_factors) == 0:
        print(f"⚠️ Skipping Abnormal_{i} - no data available.")
        continue

    print(f"✅ Plotting for Abnormal_{i}")

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    # fig.suptitle(f"Normal vs Abnormal_{i} - A, B, C", fontsize=16)

    variables = ['A', 'B', 'C']

    for idx in range(3):
        # Plot Normal
        axs[idx, 0].imshow(test_factors_normal[0][idx], label='Normal', aspect='auto')
        axs[idx, 0].set_title(f'Normal - {idx}')

        # Plot Abnormal
        axs[idx, 1].imshow(abnormal_factors[0][idx], label=f'Abnormal_{i}', aspect='auto' )
        axs[idx, 1].set_title(f'Abnormal_{i} - {idx}')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
