
import matplotlib.pyplot as plt

def plot_training_metrics(agent):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(agent.training_loss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(agent.training_accuracy)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()
