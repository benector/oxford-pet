import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from data.custom_dataset import OxfordPetDataset
import numpy as np


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # mesmo tamanho que você usa no treino
    transforms.ToTensor()
])

dataset = OxfordPetDataset(
    root='./data',
    transform=transform
)


def compute_mean_std(dataset, batch_size=12, num_workers=1):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)

    for data, _ in loader:
        # data shape: (B, C, H, W)
        b, c, h, w = data.shape
        n_pixels += b * h * w

        channel_sum += data.sum(dim=[0, 2, 3])  # soma de cada canal
        channel_squared_sum += (data ** 2).sum(dim=[0, 2, 3])  # soma dos quadrados

    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_squared_sum / n_pixels - mean ** 2)

    return mean.tolist(), std.tolist()

print(compute_mean_std(dataset))
