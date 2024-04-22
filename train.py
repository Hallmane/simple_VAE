import argparse
import torch
import torchvision.datasets as datasets
from tqdm import tqdm # progressbar
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms # augmentations
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import os
import numpy as np

from audio_vae import AudioVAE

# vae config
INPUT_DIM = (1, 256, 256)
LATENT_DIM = 64
CONV_FILTERS=[64, 128, 256, 512]
CONV_KERNEL_SIZES=[3,3,3,3]
CONV_PADDING=[0,0,0,0]
CONV_STRIDES=[2,2,2,2]
ACTIVATION_FUNCTION=nn.LeakyReLU
BATCH_NORM=True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="audio_vae_final_checkpoint.pth" ,help="Path to the saved model weights")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=int, default=1e-04)
    parser.add_argument("--spectr_path", type=str, default="dataset/spectrograms/")
    return parser.parse_args()

def main():
    args = parse_args()


    model = AudioVAE(
        INPUT_DIM, 
        LATENT_DIM, 
        CONV_FILTERS, 
        CONV_KERNEL_SIZES, 
        CONV_PADDING, 
        CONV_STRIDES, 
        ACTIVATION_FUNCTION, 
        BATCH_NORM
    ).to(DEVICE)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        print(f"starting from previous run...")


    # model, optim and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    x_train = load_spectrograms(args.spectr_path)
    train_dataset = TensorDataset(x_train, x_train)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train(model, train_loader, args.epochs, optimizer)
    visualize_reconstruction(model, train_loader)


def load_spectrograms(path, size=(256, 256)):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
    data = [np.load(file) for file in files if np.load(file).shape == size]
    data = np.expand_dims(data, axis=1)
    return torch.tensor(data, dtype=torch.float32).to(DEVICE)

def train(model, train_loader, num_epochs, optimizer):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            x_reconstructed, mu, var = model(inputs)
            loss = model._combined_loss(x_reconstructed, targets, mu, var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        torch.save(model.state_dict(), 'audio_vae_final_checkpoint.pth')


def visualize_reconstruction(model, loader):
    model.eval()
    inputs, _ = next(iter(loader))
    with torch.no_grad():
        reconstructed, _, _ = model(inputs.to(DEVICE))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(inputs[0][0].cpu(), cmap='viridis')
    axs[0].set_title('Original')
    axs[1].imshow(reconstructed[0][0].cpu(), cmap='viridis')
    axs[1].set_title('Reconstructed')
    plt.show()

if __name__ == "__main__":
    main()