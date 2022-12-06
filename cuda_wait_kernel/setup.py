from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_wait_cuda',
    ext_modules=[
        CUDAExtension('custom_wait_cuda', [
            'custom_wait_cuda.cpp',
            'custom_wait_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
