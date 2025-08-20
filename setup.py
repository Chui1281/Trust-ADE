from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trust-ade",
    version="2.0.0",
    author="Trust-ADE Contributors",
    author_email="contact@trust-ade.org",
    description="Advanced Model Evaluation with CUDA Support and Trust-ADE Protocol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Trust-ADE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0", "mypy>=1.0.0"],
        "cuda": ["torch>=2.0.0+cu118"],
        "xanfis": ["xanfis>=1.0.0", "mealpy>=3.0.0", "permetrics>=1.4.0"],
        "full": ["torch>=2.0.0", "xanfis>=1.0.0", "mealpy>=3.0.0", "permetrics>=1.4.0"],
    },
    entry_points={
        "console_scripts": [
            "trust-ade=trust_ade.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "trust_ade": ["*.yml", "*.yaml", "*.json"],
    },
)
