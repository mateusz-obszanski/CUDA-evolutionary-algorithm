#!/usr/bin/env sh
nvcc ./sketch-cuda.cu -o ./build/sketch-cuda --std=c++20 -O0 -g -G --include-path include --extended-lambda
