#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='d2c',
    description="D2C is a Data-driven Control Library based on reinforcement learning.",
    url='https://github.com/EzioPeter/D2C-XJTU.git',
    python_requires=">=3.11",
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium==0.28.1',
        'numpy',
        'pandas',
        'torch==2.4.0',
        'tqdm',
        'json5',
        'wandb',
        'tensorboard',
        'easydict',
        'h5py',
        'fire',
        'tyro',
        'ml_collections==0.1.1',
        'Cython==0.29.33',
        'mujoco-py==2.1.2.14',
    ],
)
