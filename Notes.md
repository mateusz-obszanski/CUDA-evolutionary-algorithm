# Development notes

## Requirements

- CUDA toolkit >= 12
- C++ standard 20
- clang
- cmake
- ninja-build

## Git

- each new feature/bugfix on a separate branch
- squash commits before merge

## Conventions

- headers: *.hxx
- (eventually) source: *.cxx
- common header for types
- common style in `.clang-format` file
- `#pragma once` instead of include guards
- cudaMalloc/Free warpped in RAII
- relative `#include` paths
- in-development files should be named `playground.<extension> | *.playground.* | *.todo.*`

## Misc

- if unsure, use `std::size_t` as indexing type
- invoke `cmake --build .` inside `build` directory
- format all files in project:

```bash
find . -iname '*.cxx' -o -iname '*.hxx' -o -iname '*.cu' -o -iname '*.cuh' | xargs clang-format -i
```

## Discussion points

- CUDA Thrust library - why not use device_vector wrapped with 2D or 3D adapter?
  - no option for cudaMalloc2D/3D and similar

## Ideas

- functional:
  - `transformZip2` for crossover
  - `transform` overload with `begin` and `end` pseudo-iterators as `swapRange` engine,
    for swapping-crossover
  - `swapRange` `__device__` function for iterative reuse for multiple swap ranges during crossover
