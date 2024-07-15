from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_features, num_classes) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_features, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        epochs = config.get("local_epochs", 1)
        
        # Debug: Print learning rate and momentum
        print(f"Training with lr: {lr}, momentum: {momentum}, epochs: {epochs}")

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)
        
        # Debug: Evaluate on the training set after training
        train_loss, train_accuracy = test(self.model, self.trainloader, self.device)
        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        
        # Debug: Print evaluation results
        print(f"Evaluation loss: {loss}, Evaluation accuracy: {accuracy}")

        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}

def generate_client_fn(trainloaders, valloaders, num_features, num_classes):
    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_features=num_features,
            num_classes=num_classes,
        ).to_client()
    return client_fn
