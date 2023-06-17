#!/usr/bin/env bash
# clang++ dev.cxx -o dev \
#   --std=c++20 -iquoteinclude -O3 -Wall -Wextra -Wpedantic $@

nvcc dev.cxx -o dev --std=c++20 --include-path include -O3 

