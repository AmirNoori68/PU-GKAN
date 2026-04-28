# ==========================================================
# model_gkan.py
#
# Gaussian KAN (GKAN)
#
# Input data should be mapped to [0, 1]^d before calling
# the model. Centers are uniformly placed in [0, 1].
# ==========================================================

import torch
import torch.nn as nn
import numpy as np


class GKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_grid, eps=0.1, device="cpu"):
        super().__init__()

        self.inputdim = int(input_dim)
        self.outdim = int(output_dim)
        self.num_grid = int(num_grid)
        self.eps = float(eps)

        self.coeffs = nn.Parameter(
            torch.empty(self.inputdim, self.outdim, self.num_grid, device=device)
        )

        nn.init.normal_(
            self.coeffs,
            mean=0.0,
            std=1.0 / (self.inputdim * self.num_grid),
        )

        self.register_buffer(
            "centers",
            torch.linspace(
                0.0,
                1.0,
                self.num_grid,
                device=device,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(self, x):
        xg = x.reshape(-1, self.inputdim, 1).expand(-1, -1, self.num_grid)

        r2 = (xg - self.centers.reshape(1, 1, self.num_grid)) ** 2
        phi = torch.exp(-r2 / (self.eps ** 2))

        return torch.einsum("nig,iog->no", phi, self.coeffs)

    def set_eps(self, new_eps):
        self.eps = float(new_eps)


class GKAN(nn.Module):
    def __init__(self, a, *, num_grid, eps=0.1, device="cpu"):
        super().__init__()

        num_layers = len(a) - 1

        if isinstance(eps, (list, tuple, np.ndarray)):
            assert len(eps) == num_layers
            eps_list = [float(e) for e in eps]
        else:
            eps_list = [float(eps)] * num_layers

        self.layers = nn.ModuleList([
            GKANLayer(
                input_dim=i,
                output_dim=j,
                num_grid=num_grid,
                eps=eps_list[k],
                device=device,
            )
            for k, (i, j) in enumerate(zip(a[:-1], a[1:]))
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_with_layer_inputs(self, x):
        z_ins, z_outs = [], []

        z = x
        for layer in self.layers:
            z_ins.append(z)
            z = layer(z)
            z_outs.append(z)

        return z_ins, z_outs

    def get_eps(self):
        return [layer.eps for layer in self.layers]

    def set_eps(self, eps_list):
        assert len(eps_list) == len(self.layers)
        for layer, e in zip(self.layers, eps_list):
            layer.set_eps(e)