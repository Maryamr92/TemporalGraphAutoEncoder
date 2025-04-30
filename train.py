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
        epochs=500,
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

    # Reset model weights like you're wiping a whiteboard
    if hasattr(model, 'encoder') and isinstance(model.encoder, nn.ModuleList):
        for layer in model.encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'reset_parameters'):
        model.decoder.reset_parameters()

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
