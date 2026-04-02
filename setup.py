from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="mtopic",
    version="1.1",
    python_requires=">=3.9",
    install_requires=["numpy==1.26.1",
                      "scipy==1.11.3",
                      "joblib==1.3.2",
                      "tqdm",
                      "scikit-learn==1.3.0",
                      "matplotlib",
                      "umap-learn==0.5.6",
                      "pandas==2.2.2",
                      "kneed==0.8.5",
                      "muon",
                      "torch",
                      "ipykernel",
                      "ipywidgets"],
    author="Piotr Rutkowski",
    author_email="prutkowski@ichf.edu.pl",
    description="Multimodal single-cell topic modeling",
    license="BSD-Clause 2",
    keywords=[
        "single-cell",
        "multimodal", 
        "multiomics",],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",],
    packages=find_packages(),
)
