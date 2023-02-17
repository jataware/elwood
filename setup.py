#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup


def read_requirements(path: str):
    with open(path) as f:
        return f.read().splitlines()


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

install_requirements = read_requirements("requirements.txt")

setup(
    author="Brandon Rose, Powell Fendley"
    author_email="brandon@jataware.com, powell@jataware.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="An open source dataset transformation, standardization, and normalization python library.",
    entry_points={
        "console_scripts": [
            "elwood=elwood.cli:cli",
        ],
    },
    setup_requires=["setuptools<58.0.0"],
    install_requires=install_requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="elwood",
    name="elwood",
    package_data={"elwood": ["data/*"]},
    packages=find_packages(include=["elwood", "elwood.*"]),
    test_suite="tests",
    url="https://github.com/jataware/elwood",
    version="0.1.0",
)
