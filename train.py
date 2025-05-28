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
        input_tensor,
        A, B, C,
        Rank,
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

    # # Reset model weights like you're wiping a whiteboard
    # if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
    #     for layer in model.layers:
    #         if hasattr(layer, 'reset_parameters'):
    #             layer.reset_parameters()
    #
    # if hasattr(model, 'decoder') and hasattr(model.decoder, 'reset_parameters'):
    #     model.decoder.reset_parameters()

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
        optimizer.zero_grad()



        # Fire the neurons!
        A_hat, B_hat, C_hat = model(input_tensor, A, B, C, Rank)

        # Cast the loss spell
        loss_A = criterion(A_hat, A)
        loss_B = criterion(B_hat, B)
        loss_C = criterion(C_hat, C)
        train_loss = loss_A + loss_B + loss_C

        # Backprop like a boss
        train_loss.backward()
        grad_history.append(get_gradients(model))
        optimizer.step()

        train_losses.append(train_loss.item())
        if verbose:
            print(f"[Epoch {epoch + 1}] Training Loss: {train_loss.item():.12f}")

        # Evaluate without breaking the space-time continuum
        model.eval()
        with torch.no_grad():
            A_hat, B_hat, C_hat = model(input_tensor, A, B, C, Rank)
            val_loss = criterion(A_hat, A) + criterion(B_hat, B) + criterion(C_hat, C)
            val_losses.append(val_loss.item())
            if verbose:
                print(f"[Epoch {epoch + 1}] Validation Loss: {val_loss.item():.12f}")

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

    return train_losses, val_losses, grad_history, A_hat, B_hat, C_hat


def evaluate_model(model, input_tensor, A_true, B_true, C_true, Rank, verbose=False):
    """
    Evaluate a trained TGCN model on given inputs.

    Args:
        model (nn.Module): Trained TGCN_Autoencoder model
        input_tensor (torch.Tensor): Input data for evaluation
        A_true, B_true, C_true (torch.Tensor): Ground-truth factor matrices
        Rank (int): Rank used in tensor decomposition
        verbose (bool): Whether to print evaluation metrics

    Returns:
        Tuple: (A_pred, B_pred, C_pred, loss_dict)
    """
    model.eval()
    criterion = nn.MSELoss()

    # A_true, B_true, C_true = adj_list_paraf

    with torch.no_grad():
        A_pred, B_pred, C_pred = model(input_tensor, A_true, B_true, C_true, Rank)

        loss_A = criterion(A_pred, A_true).item()
        loss_B = criterion(B_pred, B_true).item()
        loss_C = criterion(C_pred, C_true).item()
        total_loss = loss_A + loss_B + loss_C

        if verbose:
            print("=== Evaluation Results ===")
            print(f"MSE Loss A: {loss_A:.6f}")
            print(f"MSE Loss B: {loss_B:.6f}")
            print(f"MSE Loss C: {loss_C:.6f}")
            print(f"Total Loss: {total_loss:.6f}")

            print(f"\nNorm(A_error): {torch.norm(A_pred - A_true):.6f}")
            print(f"Norm(B_error): {torch.norm(B_pred - B_true):.6f}")
            print(f"Norm(C_error): {torch.norm(C_pred - C_true):.6f}")

    return total_loss, A_pred, B_pred, C_pred

