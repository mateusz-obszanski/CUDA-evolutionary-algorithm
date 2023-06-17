#!/usr/bin/env bash
# clang++ dev.cxx -o dev \
#   --std=c++20 -iquoteinclude -O0 -g -Wall -Wextra -Wpedantic $@

# gcc dev.cxx -o dev \
#   --std=c++20 -iquoteinclude -O0 -g -Wall -Wextra -Wpedantic $@

# clang++-15 dev.cxx -o dev \
#   --std=c++20 -iquoteinclude -O0 -g -Wall -Wextra -Wpedantic $@

# This works somehow
nvcc dev.cxx -o ./build/dev-test --std=c++20 -O0 -g -G --include-path include
