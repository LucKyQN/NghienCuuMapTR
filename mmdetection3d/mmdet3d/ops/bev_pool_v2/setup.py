from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_pool_v2_ext',
    ext_modules=[
        CUDAExtension('bev_pool_v2_ext', ['bev_pool_v2.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
