from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

fwht_extension = CUDAExtension(
    'fwht_cuda', [
        'fwht_cuda.cpp',
        'fwht_cuda_kernel.cu',
    ]
)

setup(name='fwht_cuda', ext_modules=[fwht_extension], cmdclass={'build_ext': BuildExtension})
