#!/bin/bash

# Check versions
pip --version
python --version

# Install sentencepiece
pip install -U pip setuptools wheel
pip install sentencepiece

# Check versions
pip --version
python --version

# Install requirements
pip install -r requirements.txt