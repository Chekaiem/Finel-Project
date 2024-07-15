from typing import Dict
from collections import OrderedDict
import flwr as fl
from omegaconf import DictConfig

import torch

from model import Net, test


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)
        
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_features, num_classes, testloader):
    """Return an evaluation function for centralized evaluation."""    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):        
        model = Net(num_features, num_classes)  
         # Pass both num_features and num_classes
        
        params_dict = zip(model.state_dict().keys(), parameters)    
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)        
        loss, accuracy = test(model, testloader, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))        
        return float(loss), {"accuracy": float(accuracy)}
    
        
    return evaluate
    