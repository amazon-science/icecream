# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os

from setuptools import find_packages, setup

setup(
    name="ICECREAM",
    version="1.0",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=['numpy',
                      'networkx'],
    root_script_source_version="default-only",
)