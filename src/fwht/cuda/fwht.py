from torch.autograd import Function
import fwht_cuda


class FWHTFunction(Function):
    """
    The un-normalized Hadamard transform (i.e. without dividing by sqrt(2)).
    """

    @staticmethod
    def forward(ctx, x):
        return fwht_cuda.fwht(x)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHTFunction.apply(grad_output)
