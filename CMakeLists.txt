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
  message("Detected preferred compiler: clang++")
  set(CMAKE_CUDA_HOST_COMPILER ${clangxx_path})
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
  message("Unknown host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# ninja instead of make
find_program(ninja_path ninja)
if(ninja_path)
  message("Detected preferred build tool: Ninja")
  set(CMAKE_GENERATOR Ninja)
else()
  message("Detected build tool: ${CMAKE_GENERATOR}")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
)

link_directories(
  ${CUDA_LIBRARY_DIRS}
)

# control where the static and shared libraries are built so that on windows
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

message("archive output directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
message("library output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message("runtime output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

add_executable(EA src/EA.cu)
# vvv for recursive kernel calls
set_property(TARGET EA PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(EA PUBLIC cxx_std_20)

target_compile_options(
  EA PRIVATE
  # CUDA compiler flags
  # --verbose
  -O3
  --forward-unknown-opts
  --extended-lambda
  ${host_compiler_flags}
)
