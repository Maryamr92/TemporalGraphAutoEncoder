from Data_Generator import generate_temporal_graph_dataset
from Model import TGCN_Autoencoder
from model_summary import prepare_wrapped_model, show_model_summary
import torch
from torchinfo import summary
from train import train_model, evaluate_model
from train_batch import train_model as train_model_batch
from visulisation import plot_training_curves
from parafac_fun import parafac_gen, parafac_gen_list
from preparing_data import split_dataset

### ==== 1. preparing dataset

nNodes = 20
Time = 12
Features = 1
Rank = 3
stochastic_data = True
save_path = f"Generated_Data/temporal_graph_dataset_{Features}-{nNodes}-{Time}_.pt"
save_path_abnormal = f"Generated_Data/temporal_graph_dataset_abnormal_{Features}-{nNodes}-{Time}_.pt"
### Generate and save dataset, watch node 0 and node 4
dataset = generate_temporal_graph_dataset(
    nNodes=nNodes,
    Time=Time,
    Features=Features,
    num_cycle=3,
    num_comm=2,
    num_samples=200,
    high_prob=0.2,
    low_prob=0.03,
    save_path=save_path,
    visualize=True,
    node_i= 0,   # Choose a node within num of node set
    node_j= 2,    # Choose a node within num of node set
    num_snapshots = Time
)

print(f"Dataset saved as {save_path}")
#### ==== 2. Making the model

# Example of layer dimensions: input_dim=10, hidden layers, final output_dim=5
layer_dims = [(2,2,2)]  # input -> hidden1 -> hidden2 -> hidden3 -> hidden4 -> output
input_shape= (nNodes, Time, Features)
# Create model
model = TGCN_Autoencoder(
    input_X= input_shape,
    layer_dims=layer_dims,
    nNodes=nNodes,
    Time=Time,
    Rank=Rank,
    dropout=0.0
)

summary(model)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Wrap and prepare
wrapped_model = prepare_wrapped_model(model, nNodes=nNodes, Time=Time, Rank=Rank)

# Show model summary
show_model_summary(wrapped_model, nNodes=nNodes, Time=Time, Features=Features)


#### ==== 3. preparing the data

# choose the dataset:

if stochastic_data:

    # # Load your saved tensors
    loaded_data = torch.load(save_path)
    adj_list_full = loaded_data["adj"]
    feat_list_full = loaded_data["feat"]
    # ---- for 1 sample ----- #####
    # input_tensor = feat_list_full[0]
    # A_true, B_true, C_true = parafac_gen(adj_list_full[0], Rank)

    adj_list_train, feat_list_train, adj_list_val, feat_list_val = split_dataset(
        adj_list_full, feat_list_full, train_ratio=0.8, seed=None)

    adj_list_train = parafac_gen_list(adj_list_train, Rank)
    adj_list_val = parafac_gen_list(adj_list_val, Rank)

else:
    # Define A, B, and C
    A_true = torch.randn(nNodes, Rank, dtype=torch.float)
    B_true = torch.randn(nNodes, Rank, dtype=torch.float)
    C_true = torch.randn(Time, Rank, dtype=torch.float)
    input_tensor = torch.randn(nNodes, Time, Features, dtype=torch.float)

### ==== 4. train and test the data

# train_losses, val_losses, grad_history, A_hat, B_hat, C_hat = train_model(
#     model=model,
#     input_tensor=input_tensor,
#     A=A_true, B=B_true, C=C_true,
#     Rank=Rank,
#     epochs=1000,
#     learning_rate=5e-3,
#     patience=10,
#     verbose=True,
#     use_early_stopping=False
# )


### ==== 4. train and test the data

train_losses, val_losses, grad_history =  train_model_batch(
        model = model,
        adj_list_train = adj_list_train,
        feat_list_train = feat_list_train,
        adj_list_val =  adj_list_val,
        feat_list_val = feat_list_val,
        Rank = Rank,
        batch_size= int(len(adj_list_train)),
        # batch_size= 32,
        epochs=1000,
        learning_rate=5*1e-3,
        patience=15,
        min_delta=1e-4,
        verbose=True,
        use_early_stopping=False)

#### ==== 5. Visualization

plot_training_curves(train_losses, val_losses, log_axis='False')
#
# print(f'norm(A_hat - A_true), {torch.norm(A_hat - A_true)}')
# print(f'norm(B_hat - B_true), {torch.norm(B_hat - B_true)}')
# print(f'norm(C_hat - C_true), {torch.norm(C_hat - C_true)}')
#
#

### ===== 6. Evaluation

abnormal_data = generate_temporal_graph_dataset(
    nNodes=nNodes,
    Time=Time,
    Features=Features,
    num_cycle=2,
    num_comm=2,
    num_samples=1,
    high_prob=0.03,
    low_prob=0.2,
    save_path=save_path_abnormal,
    visualize=False,
    node_i= 0,   # Choose a node within num of node set
    node_j= 2,    # Choose a node within num of node set
    num_snapshots = Time
)
abnormal_data_ = torch.load(save_path_abnormal)
adj_list_full_mal = abnormal_data_["adj"]
feat_list_full_mal = abnormal_data_["feat"]

input_tensor = feat_list_full_mal[0]
A_true_mal, B_true_mal, C_true_mal = parafac_gen(adj_list_full_mal, Rank)
# After training
A_pred_mal, B_pred_mal, C_pred_mal, eval_metrics = evaluate_model(
    model, input_tensor, A_true_mal, B_true_mal, C_true_mal, Rank
)

