from Data_Generator import generate_temporal_graph_dataset
from Model import TGCN_Autoencoder
from model_summary import prepare_wrapped_model, show_model_summary
import torch
from torchinfo import summary
from train import train_model
from visulisation import plot_training_curves
from parafac_fun import parafac_gen, parafac_gen_list

### ==== 1. preparing dataset

nNodes = 5
Time = 10
Features = 1
Rank = 3

# Generate and save dataset, watch node 0 and node 4
dataset = generate_temporal_graph_dataset(
    nNodes=nNodes,
    Time=Time,
    Features=Features,
    num_blocks=2,
    num_samples=1,
    high_prob=0.9,
    low_prob=0.05,
    save_path="temporal_graph_dataset1_5_10_1.pt",
    visualize=False,
    node_i= 0,   # Choose a node within num of node set
    node_j= 4    # Choose a node within num of node set
)

#### ==== 2. Making the model

# Example of layer dimensions: input_dim=10, hidden layers, final output_dim=5
layer_dims = [(3,2,2)]  # input -> hidden1 -> hidden2 -> hidden3 -> hidden4 -> output
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

# Load your saved tensors
loaded_data = torch.load("temporal_graph_dataset1_5_10_1.pt")
adj_list_full = loaded_data["adj"]
feat_list_full = loaded_data["feat"]

# Define A, B, and C
A_true = torch.randn(nNodes, Rank, dtype=torch.float)
B_true = torch.randn(nNodes, Rank, dtype=torch.float)
C_true = torch.randn(Time, Rank, dtype=torch.float)
input_tensor = torch.randn(nNodes, Time, Features, dtype=torch.float)

A_true, B_true, C_true = parafac_gen(adj_list_full, Rank)

#### ==== 4. train and test the data

train_losses, val_losses, grad_history, A_hat, B_hat, C_hat = train_model(
    model=model,
    input_tensor=input_tensor,
    A=A_true, B=B_true, C=C_true,
    Rank=Rank,
    epochs=300,
    learning_rate=5e-3,
    patience=10,
    verbose=True,
    use_early_stopping=False
)

#### ==== 5. Visualization

plot_training_curves(train_losses, val_losses, log_axis='False')

print(f'norm(A_hat - A_true), {torch.norm(A_hat - A_true)}')
print(f'norm(B_hat - B_true), {torch.norm(B_hat - B_true)}')
print(f'norm(C_hat - C_true), {torch.norm(C_hat - C_true)}')




