import matplotlib.pyplot as plt


def plot_training_curves(train_losses, val_losses, log_axis='True'):
    """Visualize your model's learning adventure over time."""
    # Plot the training loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    if log_axis:
        plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save the figure
    plt.savefig('100samples-R3-batch10.png', dpi=300)  # You can change the filename and dpi if needed

    plt.show()
