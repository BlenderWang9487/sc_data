from setuptools import find_packages, setup

short_description = "A python library to help you build huggingface datasets from anndata / sparse matrix fast."
with open("README.md", "r") as readme:
    long_description = readme.read()
packages = find_packages(exclude=["tests*"])
requires = ["scanpy", "datasets"]

setup(
    name="sc_data",
    version="0.0.1",
    packages=packages,
    license="MIT",
    description=short_description,
    long_description=long_description,
    install_requires=requires,
    author="Blender Wang",
    author_email="developinblend@gmail.com",
)
