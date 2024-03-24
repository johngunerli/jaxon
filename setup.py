from setuptools import setup, find_packages

setup(
    name="jaxon", 
    version="0.1.0",
    packages=find_packages(include=["jaxon"]),
    install_requires=["jax"],
    author="John Gunerli",
)