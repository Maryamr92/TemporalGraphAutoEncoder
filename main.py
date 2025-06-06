from Data_Generator import generate_temporal_graph_dataset
from Model import TGCN_Autoencoder
from model_summary import prepare_wrapped_model, show_model_summary
import torch
from torchinfo import summary
from train import train_model, evaluate_model
from train_batch import train_model as train_model_batch
from train_batch import evaluate_model as evaluate_model_batch
from visulisation import plot_training_curves
from parafac_fun import parafac_decomposition_list_dense, parafac_decomposition_list_sparse, parafac_decomposition_dense
from preparing_data import split_dataset, split_dataset_sparse
import matplotlib.pyplot as plt
import time

### ==== 1. preparing dataset

nNodes = 20
Time = 10
Features = 1
Rank = 2
# ======== if stochastics_data:

num_comm = 2
num_cycle = 2
num_samples = 100
use_sparse = 1

layer_dims = [8]  # input -> hidden1 -> hidden2 -> hidden3 -> hidden4 -> output

save_path = f"Generated_Data/temporal_graph_dataset_{Features}-{nNodes}-{Time}_.pt"
save_path_abnormal = f"Generated_Data/temporal_graph_dataset_abnormal_{Features}-{nNodes}-{Time}_.pt"
### Generate and save dataset, watch node 0 and node 4
# dataset = generate_temporal_graph_dataset(
#     nNodes=nNodes,
#     Time=Time,
#     Features=Features,
#     num_cycle=num_cycle,
#     num_comm=num_comm,
#     num_samples=num_samples,
#     high_prob=0.2,
#     low_prob=0.03,
#     save_path=save_path,
#     visualize=False,
#     node_i= 0,   # Choose a node within num of node set
#     node_j= 2,    # Choose a node within num of node set
#     num_snapshots = Time
# )

print(f"Dataset saved as {save_path}")
#### ==== 2. Making the model
start_time = time.time()


# Example of layer dimensions: input_dim=10, hidden layers, final output_dim=5
input_shape= (nNodes, Time, Features)
# Create model
model = TGCN_Autoencoder(
    input_X= input_shape,
    layer_dims=layer_dims,
    nNodes=nNodes,
    Time=Time,
    Rank=Rank,
    dropout=0.2
)

summary(model)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Wrap and prepare
wrapped_model = prepare_wrapped_model(model, nNodes=nNodes, Time=Time, Rank=Rank)

# Show model summary
show_model_summary(wrapped_model, nNodes=nNodes, Time=Time, Features=Features)


#### ==== 3. preparing the data

# choose the dataset:

if num_samples > 1:

    # # Load your saved tensors
    loaded_data = torch.load(save_path)
    adj_list_full = loaded_data["adj"]
    feat_list_full = loaded_data["feat"]
    adj_list_full_spars = loaded_data["adj_spr"]

    adj_list_train, feat_list_train, adj_list_val, feat_list_val = split_dataset(
        adj_list_full, feat_list_full, train_ratio=0.8, seed=42)

    # Apply decomposition based on the mode
    if use_sparse:

        adj_list_train = parafac_decomposition_list_sparse(adj_list_train, Rank)
        adj_list_val = parafac_decomposition_list_sparse(adj_list_val, Rank)

    else:

        adj_list_train = parafac_decomposition_list_dense(adj_list_train, Rank)
        adj_list_val = parafac_decomposition_list_dense(adj_list_val, Rank)


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
#
# print(f'norm(A_hat - A_true), {torch.norm(A_hat - A_true)}')
# print(f'norm(B_hat - B_true), {torch.norm(B_hat - B_true)}')
# print(f'norm(C_hat - C_true), {torch.norm(C_hat - C_true)}')
#
#


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
        num_snapshots = Time
    )
    return abnormal_data

print('abnormal sample --------------------------')
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


for n in range(len(adj_list_val)):
    A_true_val, B_true_val, C_true_val = adj_list_val[n]
    total_loss_bon, A_pred_bon, B_pred_bon, C_pred_bon = evaluate_model(
        model, feat_list_val[n], A_true_val, B_true_val, C_true_val, Rank
    )
    results_normal += total_loss_bon

    abnormal_data = gen_abnormal_data(1)
    adj_list_full_mal = abnormal_data["adj"]
    feat_list_full_mal = abnormal_data["feat"]

    input_tensor = feat_list_full_mal[0]
    mal_factors = parafac_decomposition_list_sparse(adj_list_full_mal, Rank)
    A_true_mal, B_true_mal, C_true_mal = mal_factors[0]

    total_loss_mal, A_pred_mal, B_pred_mal, C_pred_mal = evaluate_model(
        model, input_tensor, A_true_mal, B_true_mal, C_true_mal, Rank
    )
    results_abnormal += total_loss_mal


print(f'results_abnormal_avg = {results_abnormal/len(adj_list_val)}')
print(f'results_normal_avg = {results_normal/len(adj_list_val)}')


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")

# x = np.arange(len(adj_list_val))
# width = 0.25
#
# plt.figure(figsize=(10, 6))
# plt.bar(x - width, results_abnormal, width, label='Malicious', color='tab:red')
# # plt.bar(x, results_normal, width, label='Benign', color='tab:blue')
# plt.bar(x + width, results_abnormal, width, label='Malicious', color='tab:blue')
#
#
# plt.xlabel("Run Index", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.title("Final Test Loss: Malicious vs. Benign (Each Run)", fontsize=14)
# plt.xticks(ticks=x, labels=[f"Run {i}" for i in range(len(results_normal))])
# plt.legend()
# plt.grid(True, axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("Final_Test_Loss_mal_vs_bon.png", dpi=300)
# plt.show()





# print('abnormal sample 2 --------------------------')
#
#
# print('normal sample 2--------------------------')
#
# # abnormal_data_ = torch.load(save_path_abnormal)
# adj_list_full_bon = normal_data["adj"]
# feat_list_full_bon = normal_data["feat"]
#
# input_tensor = feat_list_full_bon[1]
# A_true_bon, B_true_bon, C_true_bon = parafac_gen(adj_list_full_bon[1], Rank)
# # After training
# total_loss, A_pred_bon, B_pred_bon, C_pred_bon = evaluate_model(
#     model, input_tensor, A_true_bon, B_true_bon, C_true_bon, Rank
# )

#
#
# def main(var1, var2):
#
#     ### ==== 1. preparing dataset
#
#     Rank = var1
#
#     epochs = 1000
#     learning_rate = 5e-3
#
#     nNodes = 200
#     Time = 20
#     Features = 1
#     # Rank = 4
#
#     high_prob = 0.2
#     low_prob = 0.03
#     num_comm = 2
#     num_cycle = 2
#     num_samples = 1
#     layer_dims = var2  # input -> hidden1 -> hidden2 -> hidden3 -> hidden4 -> output
#     save_path = f"Generated_Data/temporal_graph_dataset_{Features}-{nNodes}-{Time}_.pt"
#     save_path_abnormal = f"Generated_Data/temporal_graph_dataset_abnormal_{Features}-{nNodes}-{Time}_.pt"
#
#
#     ### Generate and save dataset, watch node 0 and node 4
#     dataset = generate_temporal_graph_dataset(
#         nNodes=nNodes,
#         Time=Time,
#         Features=Features,
#         num_cycle=num_cycle,
#         num_comm=num_comm,
#         num_samples=200,
#         high_prob=high_prob,
#         low_prob=low_prob,
#         save_path=save_path,
#         visualize=False,
#         node_i=0,  # Choose a node within num of node set
#         node_j=2,  # Choose a node within num of node set
#         num_snapshots=Time
#     )
#
#     #### ==== 2. Making the model
#
#     input_shape = (nNodes, Time, Features)
#     # Create model
#     model = TGCN_Autoencoder(
#         input_X=input_shape,
#         layer_dims=layer_dims,
#         nNodes=nNodes,
#         Time=Time,
#         Rank=Rank,
#         dropout=0.0
#     )
#
#     summary(model)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # Wrap and prepare
#     wrapped_model = prepare_wrapped_model(model, nNodes=nNodes, Time=Time, Rank=Rank)
#
#     # Show model summary
#     show_model_summary(wrapped_model, nNodes=nNodes, Time=Time, Features=Features)
#
#     #### ==== 3. preparing the data
#
#     # choose the dataset:
#
#     if num_samples > 1:
#
#         # # Load your saved tensors
#         # loaded_data = torch.load(save_path)
#         adj_list_full = dataset["adj"]
#         feat_list_full = dataset["feat"]
#
#         adj_list_train, feat_list_train, adj_list_val, feat_list_val = split_dataset(
#             adj_list_full, feat_list_full, train_ratio=0.8, seed=None)
#
#         adj_list_train_paraf = parafac_gen_list(adj_list_train, Rank)
#         adj_list_val_paraf = parafac_gen_list(adj_list_val, Rank)
#
#     else:
#         ### ---- for 1 sample ----- #####
#         # loaded_data = torch.load(save_path)
#         # adj_list_full = loaded_data["adj"]
#         # feat_list_full = loaded_data["feat"]
#         adj_list_full = dataset["adj"]
#         feat_list_full = dataset["feat"]
#
#         input_tensor = feat_list_full[0]
#         A_true, B_true, C_true = parafac_gen(adj_list_full[0], Rank)
#
#     ### ==== 4. train and test the data
#
#     if num_samples == 1:
#         train_losses, val_losses, grad_history, A_hat, B_hat, C_hat = train_model(
#             model=model,
#             input_tensor=input_tensor,
#             A=A_true, B=B_true, C=C_true,
#             Rank=Rank,
#             epochs=epochs,
#             learning_rate=learning_rate,
#             patience=10,
#             verbose=True,
#             use_early_stopping=False
#         )
#
#     else:
#
#         train_losses, val_losses, grad_history = train_model_batch(
#             model=model,
#             adj_list_train=adj_list_train_paraf,
#             feat_list_train=feat_list_train,
#             adj_list_val=adj_list_val_paraf,
#             feat_list_val=feat_list_val,
#             Rank=Rank,
#             batch_size=int(len(adj_list_train_paraf)),
#             epochs=epochs,
#             learning_rate=learning_rate,
#             patience=15,
#             min_delta=1e-4,
#             verbose=True,
#             use_early_stopping=False)
#
#     ### ===== 5. Evaluation on mal data
#     print('abnormal test set -----------------------------------------')
#
#     abnormal_data = generate_temporal_graph_dataset(
#         nNodes=nNodes,
#         Time=Time,
#         Features=Features,
#         num_cycle=num_cycle,
#         num_comm=num_comm,
#         num_samples=num_samples,
#         high_prob=low_prob,
#         low_prob=high_prob,
#         save_path=save_path_abnormal,
#         visualize=False,
#         node_i=0,  # Choose a node within num of node set
#         node_j=2,  # Choose a node within num of node set
#         num_snapshots=Time
#     )
#     # abnormal_data_ = torch.load(save_path_abnormal)
#     adj_list_full_mal = abnormal_data["adj"]
#     feat_list_full_mal = abnormal_data["feat"]
#     adj_list_full_mal_paraf = parafac_gen_list(adj_list_full_mal, Rank)
#
#     # After training
#
#     if num_samples == 1:
#         test_loss_mal = evaluate_model(
#             model, feat_list_full_mal[0], adj_list_full_mal_paraf[0], Rank, verbose=True)
#
#     else:
#         test_loss_mal = evaluate_model_batch(
#         model, feat_list_full_mal, adj_list_full_mal_paraf, Rank, verbose=True)
#
#     ### ====== 6. Evaluation on bon data
#     print('normal test set ------------------------------------------------ ')
#
#     normal_data = generate_temporal_graph_dataset(
#         nNodes=nNodes,
#         Time=Time,
#         Features=Features,
#         num_cycle=num_cycle,
#         num_comm=num_comm,
#         num_samples=num_samples,
#         high_prob=high_prob,
#         low_prob=low_prob,
#         save_path=save_path_abnormal,
#         visualize=False,
#         node_i=0,  # Choose a node within num of node set
#         node_j=2,  # Choose a node within num of node set
#         num_snapshots=Time
#     )
#     # abnormal_data_ = torch.load(save_path_abnormal)
#     adj_list_full_bon = normal_data["adj"]
#     feat_list_full_bon = normal_data["feat"]
#     adj_list_full_bon_paraf = parafac_gen_list(adj_list_full_bon, Rank)
#
#     # After training
#     if num_samples==1:
#         test_loss_bon = evaluate_model(
#         model, feat_list_full_bon[0], adj_list_full_bon_paraf[0], Rank, verbose=True)
#
#     else:
#         test_loss_bon = evaluate_model_batch(
#             model, feat_list_full_bon, adj_list_full_bon_paraf, Rank, verbose=True)
#
#     return train_losses, val_losses, test_loss_mal, test_loss_bon
#
# rank_list = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# latent_space_list = [[120],[120],[120],[120],[120],[120],[120],[120],[120],[120]]
# results = {}
#
# # Use index to ensure unique keys
# for idx, (n, latent_space) in enumerate(zip(rank_list, latent_space_list)):
#     print(f"Running for Rank = {n}, latent_space = {latent_space}")
#
#     train_loss, val_loss, test_loss_mal, test_loss_bon = main(n, latent_space)
#     results[idx] = {
#         "rank": n,
#         "latent_space": latent_space,
#         "train": train_loss,
#         "val": val_loss,
#         "test_mal": test_loss_mal,
#         "test_bon": test_loss_bon
#     }
#
# # --- Plot Training & Validation Loss ---
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.style.use("ggplot")
# colors = [
#     'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
# ]
#
# plt.figure(figsize=(10, 6))
# for i in range(len(results)):
#     losses = results[i]
#     plt.plot(losses["train"], color=colors[i], label=f"Train (Run {i})")
#     plt.plot(losses["val"], color=colors[i], linestyle='--', label=f"Val (Run {i})")
#
# plt.xlabel("Epochs", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.yscale('log')
# plt.title("Training and Validation Loss (Each Run)", fontsize=14)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("Train_Val_loss_vs_nodes.png", dpi=300)
# plt.show()
#
# # --- Plot Test Loss: Malicious vs Benign ---
# mal_losses = [results[i]["test_mal"] for i in range(len(results))]
# bon_losses = [results[i]["test_bon"] for i in range(len(results))]
#
# x = np.arange(len(results))
# width = 0.25
#
# plt.figure(figsize=(10, 6))
# plt.bar(x - width/2, mal_losses, width, label='Malicious', color='tab:red')
# plt.bar(x + width/2, bon_losses, width, label='Benign', color='tab:blue')
#
# plt.xlabel("Run Index", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.title("Final Test Loss: Malicious vs. Benign (Each Run)", fontsize=14)
# plt.xticks(ticks=x, labels=[f"Run {i}" for i in range(len(results))])
# plt.legend()
# plt.grid(True, axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("Final_Test_Loss_mal_vs_bon.png", dpi=300)
# plt.show()
