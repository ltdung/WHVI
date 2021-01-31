from torch.autograd import Function
import fwht_cuda


class FWHTFunction(Function):
    """
    Python frontend for batched FWHT on the GPU (CUDA).
    """

    @staticmethod
    def forward(ctx, x):
        return fwht_cuda.fwht(x)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHTFunction.apply(grad_output)
