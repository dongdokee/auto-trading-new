#!/usr/bin/env python3
"""
Setup configuration for AutoTrading System
자동매매 시스템 설정 스크립트
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autotrading-system",
    version="1.0.0",
    author="AutoTrading Team",
    description="Korean Cryptocurrency Futures Automated Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.2",
            "pytest-asyncio>=0.23.8",
            "pytest-mock>=3.14.0",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "isort>=5.13.2",
            "flake8>=7.1.1",
            "mypy>=1.11.1",
            "pre-commit>=3.8.0",
        ],
        "docs": [
            "sphinx>=7.4.7",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autotrading=scripts.run_trading:main",
            "autotrading-backtest=scripts.backtest:main",
            "autotrading-paper=scripts.paper_trading:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)