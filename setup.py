"""Minimal setup.py so ``pip install -e .`` works."""

from setuptools import setup, find_packages

setup(
    name="ragevals",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0",
        "python-dotenv>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "ragevals=ragevals.cli:main",
        ],
    },
    python_requires=">=3.10",
)
