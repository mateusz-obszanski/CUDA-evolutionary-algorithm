#!/usr/bin/env bash
clang++ dev.cxx -o dev \
  --std=c++20 -iquoteinclude -O0 -g -Wall -Wextra -Wpedantic $@

