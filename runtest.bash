#!/bin/bash

make

./lgp

source venv/bin/activate

python3 DEAP/benchmark.py
