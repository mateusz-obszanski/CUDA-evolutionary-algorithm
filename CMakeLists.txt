cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(EA VERSION 0.1 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75) # add more if needed
endif()

# if not in PATH, specify full path to compiler executable
# uncomment to choose compiler
# set(CMAKE_CUDA_HOST_COMPILER g++)

find_program(is_clangxx_available clang++)
if(is_clangxx_available)
  set(CMAKE_CUDA_HOST_COMPILER clang++)
endif()

find_program(is_gxx_available g++)
find_program(is_msvc_available msvc.exe)

# ninja instead of make
find_program(is_ninja_available ninja)
if(is_ninja_available)
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

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

if(is_clangxx_available)
  message("Setting up for clang++")
  target_compile_options(
    EA PRIVATE
    # CUDA compiler flags
    --forward-unknown-opts
    -Xptxas
    -O3
    --extended-lambda
    --compiler-bindir=clang++

    -std=c++20
    -Wall
    -Wextra
    -Wformat=2
    -Wunused
  )
elseif(is_gxx_available)
  message("Setting up for g++")
  target_compile_options(
    EA PRIVATE
    # CUDA compiler flags
    --forward-unknown-opts
    -Xptxas
    -O3
    --extended-lambda
    --compiler-bindir=g++

    -std=c++20
    -Wall
    -Wextra
    -Wformat=2
    -Wunused
  )
elseif(is_msvc_available)
  message("Setting up for msvc")
  target_compile_options(
    EA PRIVATE
    # CUDA compiler flags
    --forward-unknown-opts
    -Xptxas
    -O3
    --extended-lambda

    -std=c++20
    -W3
  )
else()
  message("Unknown host compiler")
  target_compile_options(
    EA PRIVATE
    # CUDA compiler flags
    --forward-unknown-opts
    -Xptxas
    -O3
    --extended-lambda

    -std=c++20
  )
endif()