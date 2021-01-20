import torch


def matmul_diag_left(D_diagonal, A):
    # Multiply D * A where D_diagonal is the main diagonal of a diagonal matrix D.
    return (D_diagonal * A.T).T


def matmul_diag_right(A, D_diagonal):
    return A * D_diagonal


def is_pow_of_2(x):
    return x and (not (x & (x - 1)))


def kl_normal(mu1: torch.Tensor, sd1: torch.Tensor, mu2: torch.Tensor, sd2: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence KL{d1 || d2} where d1 and d2 are univariate normal distributions.

    :param torch.Tensor mu1: mean of d1.
    :param torch.Tensor sd1: standard deviations of d1.
    :param torch.Tensor mu2: mean of d2.
    :param torch.Tensor sd2: standard deviations of d2.
    :return torch.Tensor, scalar: KL divergence KL{d1 || d2}.
    """
    return torch.log(sd2) - torch.log(sd1) + (sd1 ** 2 + (mu1 - mu2) ** 2) / (2 * sd2 ** 2) - 0.5


def kl_diag_normal(mu1: torch.Tensor, sd1: torch.Tensor, mu2: torch.Tensor, sd2: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence KL{d1 || d2} where d1 and d2 are multivariate normal distributions with diagonal
    covariance matrices.

    :param torch.Tensor mu1: vector of means for d1.
    :param torch.Tensor sd1: vector of standard deviations for d1.
    :param torch.Tensor mu2: vector of means for d2.
    :param torch.Tensor sd2: vector of standard deviations for d2.
    :return torch.Tensor, scalar: KL divergence KL{d1 || d2}.
    """
    assert mu1.size() == mu2.size() == sd1.size() == sd2.size()

    # return torch.sum(torch.stack([kl_normal(*parameters) for parameters in zip(mu1, sd1, mu2, sd2)]))
    d = len(mu1)
    kl = 0.5 * (
            torch.sum(torch.log(sd2))
            - torch.sum(torch.log(sd1))
            - d
            + torch.sum(sd1 / sd2)
            + (mu2 - mu1) @ ((mu2 - mu1) / sd2)
    )
    return kl


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
