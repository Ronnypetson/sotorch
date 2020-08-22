from setuptools import find_packages, setup
import os

def read_text(file_name: str):
    return open(os.path.join('.', file_name)).read()

setup(
    name='sotorch',
    version='1.0',
    include_package_data=True,
    description='Scipy-like optimization tools compatible with PyTorch and with automatic Jacobian and Hessian calculation',
    packages=find_packages(),
    author='Ronnypetson Souza da Silva',
    author_email='rsronnypetson4@gmail.com',
    install_requires=[
       "torch ==1.3.1",
       "scipy >= 1.3.1",
       "numpy >= 1.17.2"
   ],
    license=read_text("LICENSE.txt")
)
