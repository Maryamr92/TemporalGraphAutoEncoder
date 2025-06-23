from Data_Generator import generate_temporal_graph_dataset
from Model import TGCN_Autoencoder
from model_summary import prepare_wrapped_model, show_model_summary
import torch
from torchinfo import summary
from train import train_model, evaluate_model
from train_batch import train_model as train_model_batch
from train_batch import evaluate_model_single, evaluate_model_list
from visulisation import plot_training_curves
from parafac_fun import parafac_decomposition_list_dense, parafac_decomposition_list_sparse, parafac_decomposition_dense
from preparing_data import split_dataset
import matplotlib.pyplot as plt
import time
import os
import numpy as np

# ==== 1. Preparing Dataset Parameters ====
nNodes = 300           # Number of nodes in the graph
Time = 200             # Number of time steps (snapshots)
Features = 1          # Feature dimensions per node
Rank = 50              # Rank of the feature matrix (used if applicable)

# Stochastic graph generation parameters
num_comm = 2          # Number of communities
num_cycle = 4         # Number of repeating cycles in graph pattern
num_samples = 50      # Number of graph sequences to generate

use_sparse = False        # Whether to use sparse matrix representation

# Neural network architecture placeholder
layer_dims = [50, 50]      # Example: input layer -> hidden layers -> output

gen_new_data = True

# Paths to save generated data
save_path = f"Generated_Data/{num_samples}_temporal_graph_dataset_{Features}-{nNodes}-{Time}_.pt"
save_path_abnormal = f"Generated_Data/{num_samples}_temporal_graph_dataset_abnormal_{Features}-{nNodes}-{Time}_.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==== 2. Generate Dataset if Not Already Saved ====
# if not os.path.exists(save_path):
if gen_new_data:
    print(f"Generating new dataset and saving to: {save_path}")
    start_time_generating_Data = time.time()

    dataset = generate_temporal_graph_dataset(
        nNodes=nNodes,
        Time=Time,
        Features=Features,
        num_cycle=num_cycle,
        num_comm=num_comm,
        num_samples=num_samples,
        high_prob=0.2,         # Probability of intra-community edge
        low_prob=0.03,         # Probability of inter-community edge
        save_path=save_path,   # File path to save the generated dataset
        visualize=False,       # Set True to plot the graph evolution
        node_i=0,              # Watch interactions from node 0
        node_j=2,              # Watch interactions to node 2
        num_snapshots=Time,     # Number of temporal snapshots
        feat_means= [1.2, 2.5, 3.1, 4.0],   # change from small values to big changes
        feat_ranges= [0.2, 0.3, 0.1, 0.4],
        device=device
    )

    end_time_generating_data = time.time()
    elapsed_time = end_time_generating_data - start_time_generating_Data

    print(f"Elapsed time of generating data: {elapsed_time:.4f} seconds")

else:
    print(f"Dataset already exists at: {save_path}")



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#### ==== 2. Making the model

# Example of layer dimensions: input_dim=10, hidden layers, final output_dim=5
input_shape= (nNodes, Time, Features)
# Create model
model = TGCN_Autoencoder(
    input_X= input_shape,
    layer_dims=layer_dims,
    nNodes=nNodes,
    Time=Time,
    Rank=Rank,
    dropout=0.2,
    device=device
).to(device)

summary(model)


# Wrap and prepare
wrapped_model = prepare_wrapped_model(model, nNodes=nNodes, Time=Time, Rank=Rank, device=device)

# Show model summary
show_model_summary(wrapped_model, nNodes=nNodes, Time=Time, Features=Features, device=device)


#### ==== 3. preparing the parafac decomposition matrices

# choose the dataset:
# # Load your saved tensors

if num_samples > 1:

    # # Load your saved tensors
    loaded_data = torch.load(save_path)
    adj_list_full = loaded_data["adj"]
    feat_list_full = loaded_data["feat"]


    adj_list_train, feat_list_train, adj_list_val, feat_list_val = split_dataset(
        adj_list_full, feat_list_full, train_ratio=0.8, seed=42)

    start_time_parafac_decomposition = time.time()

    # Apply decomposition based on the mode
    if use_sparse:

        adj_list_train = parafac_decomposition_list_sparse(adj_list_train, Rank, device=device)
        adj_list_val = parafac_decomposition_list_sparse(adj_list_val, Rank, device=device)

        end_time_parafac_decomposition = time.time()
        elapsed_time = end_time_parafac_decomposition - start_time_parafac_decomposition

        print(f"Elapsed time parse parafac_decomposition : {elapsed_time:.4f} seconds")

    else:

        adj_list_train = parafac_decomposition_list_dense(adj_list_train, Rank, device=device)
        adj_list_val = parafac_decomposition_list_dense(adj_list_val, Rank, device=device)

        end_time_parafac_decomposition = time.time()
        elapsed_time = end_time_parafac_decomposition - start_time_parafac_decomposition

        print(f"Elapsed time dense parafac_decomposition: {elapsed_time:.4f} seconds")

else:
    ### ---- for 1 sample ----- #####
    loaded_data = torch.load(save_path)
    adj_list_full = loaded_data["adj"]
    feat_list_full = loaded_data["feat"]
    input_tensor = feat_list_full[0]
    adj_tensor_paraf = parafac_decomposition_dense(adj_list_full, Rank)
    A_true, B_true, C_true = adj_tensor_paraf
    print(C_true.shape)

    # Define A, B, and C
    # A_true = torch.randn(nNodes, Rank, dtype=torch.float)
    # B_true = torch.randn(nNodes, Rank, dtype=torch.float)
    # C_true = torch.randn(Time, Rank, dtype=torch.float)
    # input_tensor = torch.randn(nNodes, Time, Features, dtype=torch.float)



### ==== 4. train and test the data

start_time_training = time.time()
if num_samples == 1:
    train_losses, val_losses, grad_history, A_hat, B_hat, C_hat = train_model(
        model=model,
        input_tensor=input_tensor,
        A=A_true, B=B_true, C=C_true,
        Rank=Rank,
        epochs=2000,
        learning_rate=5e-3,
        patience=30,
        verbose=True,
        use_early_stopping=False
    )

else:

    train_losses, val_losses, grad_history =  train_model_batch(
            model = model,
            adj_list_train = adj_list_train,
            feat_list_train = feat_list_train,
            adj_list_val =  adj_list_val,
            feat_list_val = feat_list_val,
            Rank = Rank,
            batch_size= 32,                #int(len(adj_list_train)/2),
            epochs=2000,
            learning_rate=5e-3,
            patience=50,
            min_delta=1e-4,
            verbose=True,
            use_early_stopping=True)

#### ==== 5. Visualization

plot_training_curves(train_losses, val_losses, log_axis='False')
plt.show()
#
# print(f'norm(A_hat - A_true), {torch.norm(A_hat - A_true)}')
# print(f'norm(B_hat - B_true), {torch.norm(B_hat - B_true)}')
# print(f'norm(C_hat - C_true), {torch.norm(C_hat - C_true)}')
#
#
end_time_training = time.time()
elapsed_time = end_time_training - start_time_training

print(f"Elapsed time training: {elapsed_time:.4f} seconds")

### ===== 6. Evaluation

def gen_abnormal_data(num_samples):
    abnormal_data = generate_temporal_graph_dataset(
        nNodes=nNodes,
        Time=Time,
        Features=Features,
        num_cycle=num_cycle,
        num_comm=num_comm,
        num_samples= num_samples,
        high_prob=0.03,
        low_prob=0.2,
        save_path=save_path_abnormal,
        visualize=False,
        node_i= 0,   # Choose a node within num of node set
        node_j= 2,    # Choose a node within num of node set
        num_snapshots = Time,
        feat_means=[1.2, 2.5, 3.1, 4.0],
        feat_ranges=[0.2, 0.9, 0.1, 0.6],
        device=device
    )
    return abnormal_data

print('abnormal sample --------------------------------------------------------------------')
# abnormal_data_ = torch.load(save_path_abnormal)
# adj_list_full_mal = abnormal_data["adj"]
# feat_list_full_mal = abnormal_data["feat"]
#
# input_tensor = feat_list_full_mal[0]
# A_true_mal, B_true_mal, C_true_mal = parafac_gen(adj_list_full_mal[0], Rank)
# After training
# total_loss, A_pred_mal, B_pred_mal, C_pred_mal = evaluate_model(
#     model, input_tensor, A_true_mal, B_true_mal, C_true_mal, Rank
# )
# abnormal_data_ = torch.load(save_path_abnormal)
# adj_list_full_mal = abnormal_data["adj"]
# feat_list_full_mal = abnormal_data["feat"]

# input_tensor_mal = feat_list_full_mal[0]
# A_true_mal1, B_true_mal1, C_true_mal1 = parafac_gen(adj_list_full_mal[0], Rank)
# A_true_mal = A_true_bon[torch.randperm(A_true_bon.size()[0])]
# B_true_mal = B_true_bon[torch.randperm(B_true_bon.size()[0])]
# C_true_mal = C_true_bon[torch.randperm(C_true_bon.size()[0])]

results_normal = 0.0
results_abnormal = 0.0


# A_true_val, B_true_val, C_true_val = adj_list_val[0]

abnormal_losses = []
normal_losses = []

for n in range(len(adj_list_val)):

    A_true_val, B_true_val, C_true_val = adj_list_val[n]
    total_loss_bon, A_pred_bon, B_pred_bon, C_pred_bon = evaluate_model_single(
        model, feat_list_val[n], A_true_val, B_true_val, C_true_val, Rank
    )
    results_normal += total_loss_bon
    normal_losses.append(results_normal)

    abnormal_data = gen_abnormal_data(1)
    adj_list_full_mal = abnormal_data["adj"]
    feat_list_full_mal = abnormal_data["feat"]

    input_tensor = feat_list_full_mal[0]
    mal_factors = parafac_decomposition_list_dense(adj_list_full_mal, Rank)
    A_true_mal, B_true_mal, C_true_mal = mal_factors[0]

    total_loss_mal, A_pred_mal, B_pred_mal, C_pred_mal = evaluate_model_single(
        model, input_tensor, A_true_mal, B_true_mal, C_true_mal, Rank
    )

    results_abnormal += total_loss_mal
    abnormal_losses.append(total_loss_mal)


print(f'results_abnormal_avg = {results_abnormal/len(adj_list_val)}')
print(f'results_normal_avg = {results_normal/len(adj_list_val)}')

print(f' std_abnormalities, {np.std(abnormal_losses)}')
print(f' var_abnormalities, {np.var(abnormal_losses)}')

print(f' std_normalities, {np.std(normal_losses)}')
print(f' var_normalities, {np.var(normal_losses)}')



