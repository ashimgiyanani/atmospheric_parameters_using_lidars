#!/bin/bash --login

# create the conda environment
conda env create --file ../environment.yml --force
# conda create --name py39 --file environment.txt
conda activate py39
