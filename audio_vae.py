import torch
from torch import nn


class AudioVAE(nn.Module):
    def __init__(self, 
                input_dim, 
                latent_dim, 
                conv_filters, 
                conv_kernels, 
                conv_strides, 
                n_layers=4):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        self.conf_kernerls = conv_kernels
        self.conv_strides = conv_strides

        self.res_stack = nn.Sequential(
            nn.conv1d(input_dim)
        )

    #q(latent | input)
    def encode(self, x):
        return 0

    #p_theta(input' | latent)
    def decode(self, z):
        return 0


    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs)

    

        