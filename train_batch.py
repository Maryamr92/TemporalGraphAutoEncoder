import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_gradients(model):
    """Snag the gradients from the model like a data ninja."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = {
                'min': param.grad.min().item(),
                'max': param.grad.max().item(),
                'mean': param.grad.mean().item()
            }
    return grads

def train_model(
        model,
        adj_list_train,
        feat_list_train,
        adj_list_val,
        feat_list_val,
        Rank = 1,
        batch_size= 1,
        epochs=200,
        learning_rate=1e-3,
        patience=15,
        min_delta=1e-4,
        verbose=True,
        use_early_stopping=False
):
    """The ultimate model training experience — now with knobs and sliders!"""

    # Set up the optimization wizard and loss conjurer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.reset()

    # Track the journey
    grad_history = []
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None


    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx in range(0, len(adj_list_train), batch_size):
            optimizer.zero_grad()

            batch_adj_train = adj_list_train[batch_idx:batch_idx + batch_size]
            batch_feat_train = feat_list_train[batch_idx:batch_idx + batch_size]

            batch_loss = 0.0

            for i in range(len(batch_adj_train)):
                input_tensor = batch_feat_train[i]
                # adj_tensor = batch_adj_train[i]
                A_true, B_true, C_true = batch_adj_train[i]

                # A_true, B_true, C_true = adj_tensor[0], adj_tensor[1], adj_tensor[2],
                A_hat, B_hat, C_hat = model(input_tensor, A_true, B_true, C_true, Rank)

                loss_A = criterion(A_hat, A_true)
                loss_B = criterion(B_hat, B_true)
                loss_C = criterion(C_hat, C_true)

                batch_loss += (loss_A + loss_B + loss_C)

            # Average over batch size
            batch_loss = batch_loss / len(batch_adj_train)
            batch_loss.backward()

            # Optional: clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optional: log gradients
            grad_history.append(get_gradients(model))

            optimizer.step()
            epoch_train_loss += batch_loss.item()

        avg_train_loss = epoch_train_loss / (len(adj_list_train) / batch_size)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch + 1}] Avg Training Loss: {avg_train_loss:.12f}")

        # ====== VALIDATION PHASE ======
        model.eval()

        with torch.no_grad():
            val_loss = 0.0

            for i in range(len(adj_list_val)):
                input_tensor_val = feat_list_val[i]
                # adj_tensor_val = batch_adj_val[i]

                A_true_val, B_true_val, C_true_val = adj_list_val[i]
                A_hat_val, B_hat_val, C_hat_val = model(input_tensor_val, A_true_val, B_true_val, C_true_val, Rank)

                loss_A_val = criterion(A_hat_val, A_true_val)
                loss_B_val = criterion(B_hat_val, B_true_val)
                loss_C_val = criterion(C_hat_val, C_true_val)

                val_loss += (loss_A_val + loss_B_val + loss_C_val) / len(adj_list_val)

                # print(f'norm(A_hat - A_true), {torch.norm(A_hat_val - A_true_val)}')
                # print(f'norm(B_hat - B_true), {torch.norm(B_hat_val - B_true_val)}')
                # print(f'norm(C_hat - C_true), {torch.norm(C_hat_val - C_true_val)}')

                # epoch_val_loss += batch_loss.item()

        # avg_val_loss = epoch_val_loss / (len(adj_list_val) / batch_size_val)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch + 1}] Avg Validation Loss: {val_loss:.12f}")

        # Optional early exit strategy
        if use_early_stopping:
            if val_loss.item() < best_val_loss - min_delta:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load back the golden model, if early stopping was used
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f'norm(A_hat - A_true), {torch.norm(A_hat_val - A_true_val)}')
    print(f'norm(B_hat - B_true), {torch.norm(B_hat_val - B_true_val)}')
    print(f'norm(C_hat - C_true), {torch.norm(C_hat_val - C_true_val)}')

    print(A_hat_val, A_true_val)
    print(B_hat_val, B_true_val)
    print(C_hat_val, C_true_val)

    return train_losses, val_losses, grad_history


