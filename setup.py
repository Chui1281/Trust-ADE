"""
Настройка установки пакета Trust-ADE Protocol
"""

from setuptools import setup, find_packages
import os

# Чтение README файла
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Trust-ADE Protocol - Trust Assessment through Dynamic Explainability"

# Чтение requirements
def read_requirements(filename):
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Основные зависимости
install_requires = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
]

# Опциональные зависимости
extras_require = {
    "shap": ["shap>=0.41.0"],
    "viz": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950"
    ],
    "all": [
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tqdm>=4.64.0"
    ]
}

setup(
    name="trust-ade-protocol",
    version="1.0.0",
    author="Trust-ADE Development Team",
    author_email="trust-ade@example.com",
    description="Trust Assessment through Dynamic Explainability Protocol for AI Systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/trust-ade/trust-ade-protocol",
    project_urls={
        "Documentation": "https://trust-ade.readthedocs.io/",
        "Source": "https://github.com/trust-ade/trust-ade-protocol",
        "Tracker": "https://github.com/trust-ade/trust-ade-protocol/issues",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "trust-ade-test=examples.test_installation:generate_installation_report",
        ],
    },
    keywords=[
        "artificial intelligence", "explainable ai", "trust", "machine learning",
        "bias detection", "robustness", "fairness", "interpretability"
    ],
)
