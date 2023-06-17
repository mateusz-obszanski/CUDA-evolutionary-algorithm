#!/usr/bin/env sh
nvcc ./sketch-cuda.cu -o sketch-cuda --std=c++20 -O0 -g -G --include-path include --extended-lambda
