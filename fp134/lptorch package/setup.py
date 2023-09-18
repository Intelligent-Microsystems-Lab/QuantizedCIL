from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='lptorch',
    install_requires=['torch'],
    packages=['lptorch'],
    package_dir={'lptorch': './'},
    py_modules=['__init__','major','modules','optim', 'quant', 'functions'],
    ext_modules=[
        CUDAExtension('lptorch_cuda', [
            'bind.cpp',
            'lptorch.cu',
        ])
    ],
    cmdclass={
        'build_ext':BuildExtension
    }
)
