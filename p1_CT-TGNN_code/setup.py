"""
Setup script for CT-TGNN package.

Continuous-Time Temporal Graph Neural Networks for
Encrypted Traffic Analysis in Zero-Trust Architectures
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ct-tgnn",
    version="1.0.0",
    author="Roger Nick Anaedevha",
    author_email="ar006@campus.mephi.ru",
    description="Continuous-Time Temporal Graph Neural Networks for Encrypted Traffic Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rogerpanel/CV/tree/claude/dissertation-paper-proposal-am1QM/CT-TGNN",
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "results", "logs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.0.1",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ct-tgnn-train=training.trainer:main",
            "ct-tgnn-eval=evaluation.evaluator:main",
            "ct-tgnn-download-data=data.download_datasets:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.txt"],
    },
    zip_safe=False,
)
