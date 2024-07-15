import pickle
from pathlib import Path
import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    save_path = HydraConfig.get().runtime.output_dir

    # Prepare the dataset with optimized batch size
    trainloaders, validationloaders, testloader, num_features = prepare_dataset(cfg.num_clients, cfg.batch_size)
    print(len(trainloaders), len(trainloaders[0].dataset))

    # Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, num_features, cfg.num_classes)

    # Define your strategy with reduced fraction_fit and fraction_evaluate
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.1,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(num_features, cfg.num_classes, testloader),
    )

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    # Print history to understand its structure
    print(history)

    # Save your results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history}

    with open(results_path, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot and save the training results
    plot_history(history, save_path)

def plot_history(history, save_path):
    """Plot and save the training loss and accuracy over federated learning rounds."""
    # Extract loss and accuracy from the history
    rounds = range(len(history.losses_centralized))
    losses = [loss for loss in history.losses_centralized]
    accuracies = [acc[1] for acc in history.metrics_centralized["accuracy"]]

    # Plot the loss and accuracy on the same plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(rounds, losses, label="Centralized Loss", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second y-axis that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(rounds, accuracies, label="Centralized Accuracy", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # ensure the right y-label is not slightly clipped
    plt.title("Training Loss and Accuracy Over Rounds")
    plot_path = Path(save_path) / "training_loss_accuracy.png"
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    main()
