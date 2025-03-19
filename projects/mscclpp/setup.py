from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mscclpp_allreduce',
    ext_modules=[
        CUDAExtension('mscclpp_allreduce', [
            #'skinny_gemm/skinny_gemm.cu',
            'allreduce_cuda.cu',
            #'allreduce_cuda_kernel.cu',
        ],
        include_dirs=['/usr/local/mscclpp/include', "/opt/ompi/include"],  # Adjust this path
        library_dirs=['/usr/local/mscclpp/lib', "/opt/ompi/lib", "/usr/local/lib"],      # Adjust this path
        libraries=['mscclpp', 'mpi'],
        #extra_compile_args=['-Xarch_gfx942'],
        extra_cuda_cflags=['-arch=gfx942'],
        extra_hip_cflags=['-arch=gfx942']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 