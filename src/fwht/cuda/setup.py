from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fwht_cuda',
    ext_modules=[
        CUDAExtension('fwht_cuda', [
            'fwht_cuda.cpp',
            'fwht_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
