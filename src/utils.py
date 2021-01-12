import torch


def matmul_diag_left(D, A):
    # Multiply D * A where D is the main diagonal of a diagonal matrix.
    return (D * A.T).T
