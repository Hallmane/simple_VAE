import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=784, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_to_hid = nn.Linear(input_dim, h_dim)
        self.hid_to_mu = nn.Linear(h_dim, z_dim)
        self.hid_to_sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_to_hid = nn.Linear(z_dim, h_dim)
        self.hid_to_img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    #q(latent | input)
    def encode(self, x):
        h = self.relu(self.img_to_hid(x))
        mu, sigma = self.hid_to_mu(h), self.hid_to_sigma(h)
        return mu, sigma 

    #p_theta(input | latent)
    def decode(self, z):
        h = self.relu(self.z_to_hid(z))
        return torch.sigmoid(self.hid_to_img(h)) # mnist specific normalization (between 0 and 1)


    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma) # sample a value from the normal distribution
        z_reparametrized = mu + sigma*epsilon # turn the pair of 20D vectors into distributions (gaussians)
        x_hat = self.decode(z_reparametrized)
        return x_hat, mu, sigma

if __name__ == "__main__":
    vae = VariationalAutoEncoder()
    x = torch.randn(1, 28*28) 

    x_hat, mu, sigma = vae(x)

    print(x_hat.shape)
    print(mu.shape)
    print(sigma.shape)















