from setuptools import setup, find_packages

with open("README.rst", 'r') as file_handle:
    long_description = file_handle.read()

setup(
    name = "PC",
    version = "0.0.1",
    author = "Xinqiang Ding, Bin Zhang",
    author_email = "dingxq@mit.edu",
    description = "Potential Contrasting",
    long_description = long_description,
    long_description_content_type = "text/x-rst",
    url = "https://github.com/ZhangGroup-MITChemistry/PC.git",
    packages = find_packages(),
    install_requires=['numpy>=1.14.0',
                      'scipy>=1.1.0',
                      'torch>=1.0.0'],
    license = 'MIT',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
