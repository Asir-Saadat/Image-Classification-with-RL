from plot import plot_training_metrics
from train import train_mnist_dqn

if __name__ == "__main__":
    trained_agent = train_mnist_dqn()
    plot_training_metrics(trained_agent)