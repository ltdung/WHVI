import torch


def fwht(A: torch.Tensor):
    """
    In-place fast Walsh-Hadamard transform of matrix A.
    The implicitly-represented Walsh-Hadamard matrix H is not orthonormal.
    This function is equivalent to in-place matrix multiplication,
        i.e. fwht(A) = H * A where H.size() == A.size().
    The transform is individually applied to columns in a vectorized manner.

    This is quite slow.

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


class FWHT_diag(torch.autograd.Function):
    """
    FWHT on a diagonal matrix. Accepts the main diagonal elements instead of the whole diagonal matrix.
    """

    @staticmethod
    def transform(diagonal_elements):
        H = build_H(diagonal_elements.size(0))
        return H * diagonal_elements

    @staticmethod
    def forward(ctx, input):
        return FWHT_diag.transform(input)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHT_diag.transform(grad_output)


class FWHT(torch.autograd.Function):
    @staticmethod
    def transform(tensor):
        """
        Simple implementation of FWHT, receiving as input a torch Tensor.
        Source: https://discuss.pytorch.org/t/fast-walsh-hadamard-transform/19341

        Much faster, not in-place.
        """
        n = len(tensor)
        # result = np.copy(tensor.detach().numpy())  # transform to numpy
        result = tensor.detach().numpy()  # transform to numpy

        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    result[j, :], result[j + h, :] = result[j, :] + result[j + h, :], result[j, :] - result[j + h, :]
            h *= 2

        # result /= np.sqrt(n)  # Make H orthonormal
        return torch.from_numpy(result)  # transform back to torch

    @staticmethod
    def forward(ctx, input):
        return FWHT.transform(input)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHT.transform(grad_output)


def build_H(D):
    assert (D & (D >> 1)) == 0 and D > 0, "Error: D must be a power of two."
    if D == 1:
        return torch.tensor([[1.0]])
    submatrix = build_H(D // 2)  # Division by 2
    return torch.cat([
        torch.cat([submatrix, submatrix], dim=1),
        torch.cat([submatrix, -submatrix], dim=1)
    ], dim=0)
