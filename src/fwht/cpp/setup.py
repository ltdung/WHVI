from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fwht_cpp',
    ext_modules=[
        CppExtension('fwht_cpp', ['fwht.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
