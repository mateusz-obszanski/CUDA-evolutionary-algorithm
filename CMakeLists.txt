cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(EA VERSION 0.1 LANGUAGES CXX CUDA)

set(cxx_standard 20)

set(CMAKE_CXX_STANDARD ${cxx_standard})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("C++ standard: ${CMAKE_CXX_STANDARD}")

set(CMAKE_CUDA_STANDARD ${cxx_standard})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message("CUDA C++ standard: ${CMAKE_CUDA_STANDARD}")

set(required_minimal_cuda_architecture 75)
set(required_minimal_cuda_toolkit_version 12.0)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${required_minimal_cuda_toolkit_version})
  message(FATAL_ERROR "\
  CUDA toolkit minimum version requirement not met:\n\
  required: ${required_minimal_cuda_toolkit_version}\n\
  available: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

message("CUDA compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")

foreach(cuda_arch 75 86) # add more if needed
  if(NOT cuda_arch IN_LIST CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${cuda_arch})
  endif()
endforeach()

foreach(cuda_arch ${CMAKE_CUDA_ARCHITECTURES})
  if(cuda_arch LESS ${required_minimal_cuda_architecture})
    message("discarding CUDA architecture (required>=${required_minimal_cuda_architecture}): ${cuda_arch}")
    list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${cuda_arch})
  endif()
endforeach()

list(LENGTH CMAKE_CUDA_ARCHITECTURES n_cuda_architectures)
if(n_cuda_architectures EQUAL 0)
  message(FATAL_ERROR "no cuda architectures to compile for!")
endif()

message("CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_program(clangxx_path clang++)
find_program(gxx_path g++)
find_program(msvc_path msvc.exe)

if(clangxx_path)
  message("Detected preferred compiler: clang++")
  set(CMAKE_CUDA_HOST_COMPILER Clang)
  list(APPEND host_compiler_flags
    -Wall
    -Wextra
    -Wformat=2
    -Wunused
  )
elseif(gxx_path)
  message("Detected compiler: g++")
  set(CMAKE_CUDA_HOST_COMPILER GNU)
  list(APPEND host_compiler_flags
    -Wall
    -Wextra
    -Wformat=2
    -Wunused
  )
elseif(msvc_path)
  message("Detected compiler: msvc")
  set(CMAKE_CUDA_HOST_COMPILER MSVC)
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
message("CUDA separable compilation: ON")

target_compile_features(EA PUBLIC cxx_std_20)

# optional faster CUDA math
set(fast_math ON)

if(fast_math)
  set(nvcc_fast_flags
    -ftz=true
    -prec-div=false
    -prec-sqrt=false
    -use-fast-math
  )
endif()

set(optimization_flags
  -O3
  ${nvcc_fast_flags}
)

set(nvcc_compiler_flags
  # --verbose
  --forward-unknown-opts
  --extended-lambda
)

# set(thrust_ignored_warnings -Wno-unused-function -Wno-deprecated-builtins)
set(all_compile_flags 
  ${nvcc_compiler_flags}
  ${host_compiler_flags}
  ${thrust_ignored_warnings}
  ${optimization_flags}
)

message("compilation flags: ${all_compile_flags}")

target_compile_options(EA PRIVATE ${all_compile_flags})

# for development:
add_executable(Dev src/Dev.cu)
add_executable(Dev2 src/Dev2.cu)
set_property(TARGET Dev PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Dev2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(Dev PUBLIC cxx_std_20)
target_compile_features(Dev2 PUBLIC cxx_std_20)
target_compile_options(Dev PRIVATE
  ${nvcc_compiler_flags}
  ${host_compiler_flags}
  ${thrust_ignored_warnings}
)
target_compile_options(Dev2 PRIVATE
  ${nvcc_compiler_flags}
  ${host_compiler_flags}
  ${thrust_ignored_warnings}
)
