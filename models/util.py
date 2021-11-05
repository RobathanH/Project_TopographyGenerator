import torch
import torch.nn as nn

'''
Defines utility model layers
'''

# Allows a one-input layer to take two inputs and discard the second
class MultiInputWrapper(nn.Module):
    def __init__(self, layer):
        super(MultiInputWrapper, self).__init__()
        self.layer = layer
    
    def forward(self, x):
        # Unpack inputs
        x, z = x[0], x[1:]
        
        return self.layer(x), *z


# Sampling layer
class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()
    
    def forward(self, x):
        # unpack input
        z_mean, z_log_var = x

        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + (eps * std)
        return z


class ConvVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvVAE, self).__init__()

        self.encoder = encoder
        self.sampler = Sampler()
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var