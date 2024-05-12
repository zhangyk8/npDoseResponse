#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NPDoseResponse",
    version="0.0.2",
    author="Yikun Zhang",
    author_email="yikunzhang@foxmail.com",
    description="Nonparametric Estimation and Inference on Dose-Response Curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyk8/NPDoseResponse",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    # packages=["NPDoseResponse"],
    install_requires=["numpy >= 1.16"],
    python_requires=">=3.8",
)
