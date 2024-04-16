import torch
import torchvision.datasets as datasets
from tqdm import tqdm # progressbar
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms # augmentations
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import os
import numpy as np

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
# loss weighing 
alpha = 1 
beta = 1


# dataset stuff
#dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
#dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# model, optim and loss
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss(reduction="sum")


def load_spectrograms():
    x_train = []
    for root, _, files in os.walk.('./dataset/spectrograms/'):
        for file in files:
            file_path = os.path.join(root, file)
            spectrograms = np.load(file_path)
            x_train.append(spectrograms)

    return x_train

