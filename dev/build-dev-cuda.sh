#!/usr/bin/env sh
nvcc ./dev-gpu.cu -o dev-gpu --std=c++20 --include-path include --extended-lambda
