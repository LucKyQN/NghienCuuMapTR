from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_pool_ext',
    ext_modules=[
        CUDAExtension('bev_pool_ext', ['bev_pool.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
