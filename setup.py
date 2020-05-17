#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import re
from glob import glob

from setuptools import setup, find_packages, find_namespace_packages



with open("README.md", "r") as fh:
    long_description = fh.read()

ver_file = "src/deepsensemaking/_version.py"
with open( ver_file, "r", ) as fh:
    ver_lines = fh.read()
    ver_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    ver_match = re.search( ver_regex, ver_lines, re.MULTILINE)
    if ver_match:
        ver_str = ver_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (ver_file,))


setup(
    name             = "deepsensemaking",
    version          = "0.1.2020.5.17.2",
    package_dir      = {"": "src"},
    packages         = find_namespace_packages(
        where            = "src",
        exclude          = [
            "tests",
            "tests.*",
            "*.tests",
            "*.tests.*",
        ],
    ),
    python_requires  = ">=3.6",
    url              = "https://github.com/deepsensemaking/deepsensemaking",
    download_url     = "https://github.com/deepsensemaking/deepsensemaking/releases/download/0.1.2020.5.17.1/deepsensemaking-0.1.2020.5.17.1-py3-none-any.whl",
    description      = "deepsensemaking (dsm)",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords         = [
        "deepsensemaking",
        "deep", "sensemaking",
        "natural language processing",
        "nlp",
        "machine learning",
        "ml",
        "artificial intelligence",
        "ai",
        "signals intelligence",
        "sigint",
        "communications intelligence",
        "comint",
    ],
    install_requires = [
        "argparse>=1.1",
        "pandas>=1.0.1",
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "colorama>=0.4.3",
    ],
    license          = "Apache License 2.0",
    scripts          = glob("src/bin/*"),
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
    ],
    author           = "DeepSenseMaking Developes",
    author_email     = "deepsensemaking@gmail.com",
)
