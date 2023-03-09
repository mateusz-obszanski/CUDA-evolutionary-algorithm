cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(EA VERSION 0.1 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75) # add more if needed
endif()

find_program(clangxx_path clang++)
find_program(gxx_path g++)
find_program(msvc_path msvc.exe)

if(clangxx_path)
  message("Detected compiler: clang++")
  set(CMAKE_CUDA_HOST_COMPILER ${msvc_path})
  list(APPEND host_compiler_flags
    --compiler-bindir=clang++
    -Wall
    -Wextra
    -Wformat=2
    -Wunused
  )
elseif(gxx_path)
  message("Detected compiler: g++")
  set(CMAKE_CUDA_HOST_COMPILER ${gxx_path})
  list(APPEND host_compiler_flags
    -Wall
    -Wextra
    -Wformat=2
    -Wunused
  )
elseif(msvc_path)
  message("Detected compiler: msvc")
  set(CMAKE_CUDA_HOST_COMPILER ${msvc_path})
  list(APPEND host_compiler_flags
    -W3
  )
else()
  message("Unknown compiler")
endif()

# ninja instead of make
find_program(ninja_path ninja)
if(ninja_path)
  message("Detected build tool: Ninja")
  set(CMAKE_GENERATOR Ninja)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
)

link_directories(
  ${CUDA_LIBRARY_DIRS}
)

add_executable(EA src/EA.cu)
# set_property(TARGET EA PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(EA PUBLIC cxx_std_20)

target_compile_options(
  EA PRIVATE
  # CUDA compiler flags
  --forward-unknown-opts
  -Xptxas
  -O3
  --extended-lambda
  --std=c++20
  ${host_compiler_flags}
)