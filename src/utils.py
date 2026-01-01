import matplotlib.pyplot as plt

def plot_history(history):
    """
    Plots training history (loss and PR-AUC).
    
    Args:
        history (dict): Dictionary containing "train_loss", "test_loss", and "test_pr_auc"
    """
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss (weighted)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()

    # Plot 2: PR-AUC
    plt.subplot(1, 2, 2)
    plt.plot(history["test_pr_auc"], color="green", label="Test PR-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("PR-AUC Score")
    plt.title("Metric Evolution (Higher is better)")
    plt.legend()
    plt.grid()

    plt.show()