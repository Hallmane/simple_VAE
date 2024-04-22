import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from helpers import print_shape 

class AudioVAE(nn.Module):
    def __init__(self, 
                input_dim,  # (C, H, W)
                latent_dim, # depends on how many conv layers
                conv_filters,  # [64, 128, 256, 512]
                conv_kernel_sizes, # [3, 3, 3, 3]
                conv_padding, # [0,0,0,0] for now, not sure 
                conv_strides, # [2,2,2,2]
                activation_function=nn.LeakyReLU, 
                batch_norm=True):
        super(AudioVAE, self).__init__()
        self.input_dim = input_dim
       
        # convs for encoder
        self.encoder = self._create_conv_blocks(input_dim[0], conv_filters, conv_kernel_sizes, conv_padding, conv_strides, activation_function, batch_norm, 'compression')

        # conv layers for mean and variance
        self.mu = nn.Conv2d(conv_filters[-1], latent_dim, 1)
        self.var = nn.Sequential(
            nn.Conv2d(conv_filters[-1], latent_dim, 1), 
            nn.Softplus()
        )

        # specifing decoder convs (transpose convs)
        self.decoder = self._create_conv_blocks(latent_dim, 
                                            list(reversed(conv_filters)),
                                            list(reversed(conv_kernel_sizes)),
                                            list(reversed(conv_padding)),
                                            list(reversed(conv_strides)),activation_function, batch_norm, 'expansion')

    def _create_conv_blocks(self, in_channels, conv_filters, kernel_sizes, padding, strides, activation, batch_norm, mode): 
        blocks = nn.ModuleList()
        for i, out_channels in enumerate(conv_filters):
            if mode == 'compression': #encoder convs and parts
                conv = nn.Conv2d(in_channels, out_channels, kernel_sizes[i], strides[i], padding[i])
                blocks.append(conv)
                if batch_norm:
                    blocks.append(nn.BatchNorm2d(out_channels))
                blocks.append(activation())
            elif mode == 'expansion': #decoder transpose convs and parts
                blocks.append(activation())
                output_padding = 1 if i == len(conv_filters) - 1 else 0 
                conv =  nn.ConvTranspose2d(in_channels, out_channels, kernel_sizes[i], strides[i], padding[i], output_padding)
                blocks.append(conv)

            in_channels = out_channels
            #conv.register_forward_hook(print_shape)
        return nn.Sequential(*blocks)
            

    #q(latent | input)
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        var = self.var(h)
        return mu, var

    #p_theta(input | latent)
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, var = self.encode(x)
        var = torch.exp(0.5 * var) #log
        epsilon = torch.randn_like(var)
        z_reparametrized = mu + var*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, var

    # TODO: alpha and beta vars for controlling the input of reconstruction loss and kl div in combined loss
    def _combined_loss(self, x_reconstructed, x, mu, var):
        """combined loss of reconstruction and KL div"""
        reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x)  # standard input/output loss
        kl_div = -0.5 * torch.sum(1 + torch.log(var.pow(2)) - mu.pow(2) - var.pow(2)) # pushing the individual distributions towards a standard gaussian 
        return reconstruction_loss + kl_div

    def _test_model_e2e(self):
        """sanity check"""
        test_input = torch.randn(1, *self.input_dim)
        reconstructed, mu, var = self.forward(test_input)
        print(f"Output shapes: \n input: {test_input.shape} | reconstructed: {reconstructed.shape} \n mu: {mu.shape} \n var: {var.shape}")
        pass
