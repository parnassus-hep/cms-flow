"""Mix of utility functions specifically for pytorch.
https://github.com/rodem-hep/nu2flows
"""

from functools import partial
import torch.nn as nn


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name."""
    if isinstance(name, partial):
        return name()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "lrlu":
        return nn.LeakyReLU(0.1)
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "softmax":
        return nn.Softmax()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError("No activation function with name: ", name)


def get_nrm(name: str, outp_dim: int) -> nn.Module:
    """Return a 1D pytorch normalisation layer given a name and a output size
    Returns None object if name is none."""
    if name == "batch":
        return nn.BatchNorm1d(outp_dim)
    if name == "layer":
        return nn.LayerNorm(outp_dim)
    if name == "none":
        return None
    else:
        raise ValueError("No normalistation with name: ", name)
