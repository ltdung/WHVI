import torch


def fwht(A: torch.Tensor):
    """
    In-place fast Walsh-Hadamard transform of matrix A.
    The implicitly-represented Walsh-Hadamard matrix H is not orthonormal.
    This function is equivalent to in-place matrix multiplication,
        i.e. fwht(A) = H * A where H.size() == A.size().
    The transform is individually applied to columns in a vectorized manner.

    :param torch.Tensor A: input matrix to be transformed.
    """
    assert A.dim() == 2

    n = A.size(0)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                A[j, :], A[j + h, :] = A[j, :] + A[j + h, :], A[j, :] - A[j + h, :]
        h *= 2
    return A


def build_H(D):
    assert (D & (D >> 1)) == 0 and D > 0, "Error: D must be a power of two."
    if D == 1:
        return torch.tensor([[1.0]])
    submatrix = build_H(D // 2)  # Division by 2
    return torch.cat([
        torch.cat([submatrix, submatrix], dim=1),
        torch.cat([submatrix, -submatrix], dim=1)
    ], dim=0)
