#!/usr/bin/env python3
"""
HybridGraphDB Python Bindings - Setup Script
"""

import os
import sys
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'hybrid_graphdb_py',
        ['python_bindings.cpp', 'hybrid_graphDB_v2.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '/usr/local/include',
            '.',
        ],
        library_dirs=['/usr/local/lib', '/usr/lib'],
        libraries=['rocksdb', 'pthread', 'dl', 'z', 'bz2', 'snappy', 'lz4', 'zstd'],
        extra_compile_args=['-std=c++17', '-O3', '-fPIC'],
        language='c++'
    ),
]

setup(
    name='hybrid_graphdb_py',
    version='2.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='HybridGraphDB Python Bindings - High Performance Graph Database',
    long_description=open('README_BINDINGS.md').read() if os.path.exists('README_BINDINGS.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'pybind11',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Database',
    ],
)
