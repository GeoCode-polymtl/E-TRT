from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="E-TRT",
    version="0.1.0",
    author="Clarissa Szabo-Som and Gabriel Fabien-Ouellet",
    description="Electrical Thermal Response Test (E-TRT) simulation and inversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoCode-polymtl/E-TRT",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0,<2.0",
        "scipy>=1.16.0",
        "matplotlib>=3.10.0",
        "IPython>=9.0.0",
        "jupyter>=1.1.0",
        "notebook>=7.4.0",
        "discretize>=0.12.0",
        "simpeg>=0.25.0",
        "torch>=2.2.0",
    ],
    extras_require={
        "solvers": [
            "python-mumps>=0.3.0",
            "pydiso>=0.1.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)

