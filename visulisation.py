import torch
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, log_axis='True'):
    """Visualize your model's learning adventure over time (compatible with GPU tensors)."""

    # Convert to CPU float lists (handles lists of scalars or tensors)
    train_losses = [float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss) for loss in train_losses]
    val_losses   = [float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss) for loss in val_losses]

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)

    if str(log_axis).lower() == 'true':
        plt.yscale('log')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig('100samples-R3-batch10.png', dpi=300)
    plt.show()
