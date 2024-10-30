import torch
import torch.nn as nn

def build_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None, use_layernorm=True):
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    model = nn.Sequential(*layers)
    return model
