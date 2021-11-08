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
        z = self.sampler((z_mean, z_log_var))
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var


'''
Spectral Normalization Wrapper
From https://github.com/zhaoyuzhi/deepfillv2/blob/master/deepfillv2/network_module.py#L190
'''
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)