import torch
import torch.nn as nn


class Cosine(nn.Module):
    def __init__(self):
        """
        Cosine activation function.
        """
        super().__init__()

    def forward(self, x):
        return torch.cos(x)
