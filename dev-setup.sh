#!/bin/bash
conda env create -f environment.yml
conda activate sklearndf-develop
pre-commit install