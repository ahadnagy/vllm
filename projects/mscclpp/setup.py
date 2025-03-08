from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mscclpp_allreduce',
    ext_modules=[
        CUDAExtension('mscclpp_allreduce', [
            'allreduce_cuda.cpp',
            #'allreduce_cuda_kernel.cu',
        ],
        include_dirs=['/usr/local/mscclpp/include', "/opt/ompi/include"],  # Adjust this path
        library_dirs=['/usr/local/mscclpp/lib', "/opt/ompi/lib"],      # Adjust this path
        libraries=['mscclpp', 'mpi'],
        extra_cflags=['-D__HIP_PLATFORM_AMD__'],
        extra_cude_cflags=['D__HIP_PLATFORM_AMD__']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 