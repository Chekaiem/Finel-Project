import torch
from torch.utils.data import random_split, DataLoader, Dataset
import csv

class CSVDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header row if there is one
            self.num_features = len(header) - 1  # Number of features (excluding the label)
            for row in reader:
                features = list(map(float, row[:-1]))
                label = int(row[-1])
                self.data.append((torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)))
         
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])
        return sample
   

def get_dataset(data_path: str = "data13.csv"):
    dataset = CSVDataset(data_path)
    
    # Split the dataset into training and testing sets
    test_size = int(0.2 * len(dataset))  # 20% for testing
    train_size = len(dataset) - test_size
    trainset, testset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    return trainset, testset, dataset.num_features

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset, num_features = get_dataset("data13.csv")

    if num_partitions <= 0:
        raise ValueError("Number of partitions must be greater than 0")

    # Split the training set into partitions for federated learning
    trainloaders = []
    valloaders = []
    partition_sizes = [len(trainset) // num_partitions] * num_partitions
    remainder = len(trainset) % num_partitions
    for i in range(remainder):
        partition_sizes[i] += 1
    trainsets = random_split(trainset, partition_sizes, torch.Generator().manual_seed(2023))

    for trainset_ in trainsets:
        # Split each partition into training and validation sets
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        # Create DataLoader objects for training and validation
        trainloader = DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        trainloaders.append(trainloader)
        valloaders.append(valloader)

    # DataLoader object for testing set
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader, num_features
