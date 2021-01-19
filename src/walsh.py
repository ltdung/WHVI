import torch

from utils import is_pow_of_2


def build_H(D, device):
    assert is_pow_of_2(D)
    H = build_H_recursive(D)
    H = H.to(device)
    return H


def build_H_recursive(D):
    if D == 1:
        return torch.tensor([[1.0]])
    submatrix = build_H_recursive(D // 2)  # Division by 2
    return torch.cat([
        torch.cat([submatrix, submatrix], dim=1),
        torch.cat([submatrix, -submatrix], dim=1)
    ], dim=0)
