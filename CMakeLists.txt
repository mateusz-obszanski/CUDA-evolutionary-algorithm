cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(EA VERSION 0.1 LANGUAGES CXX CUDA)

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

add_executable(EA src/EA.cu) # TODO executable

target_compile_features(EA PUBLIC cxx_std_20)

if(is_clangxx_available)
  add_compile_options(
    # CUDA compiler flags
    --forward-unknown-opts
    -Xptxas -O3,-v

    # host/fallback compiler flags
    -O3
    -std=c++20
    -pedantic
    -pedantic-errors

    # main warnings
    -Wall
    -Wextra
    -Wpedantic
    -Wuninitialized
    -Wmissing-include-dirs
    -Wshadow
    -Wundef
    -Winvalid-pch

    # extra warnings
    -Winit-self
    -Wswitch-enum
    -Wformat=2
    -Wdouble-promotion
    -Wfloat-equal
    -Wpointer-arith
    -Wstrict-overflow=5
    -Wcast-qual
    -Wcast-align
    -Wconversion
    -Wpacked
    -Wstrict-aliasing
    -Wredundant-decls
    -Wmissing-declarations
    -Wmissing-field-initializers
    -Wwrite-strings
    -Wstack-protector
    -Wpadded
    -Winline
    -Wdisabled-optimization
    -Wzero-as-null-pointer-constant
    -Wctor-dtor-privacy
    -Wold-style-cast
    -Woverloaded-virtual

    # clang-specific
    -Werror=option-ignored
    -Warc-repeated-use-of-weak
    -Wbitfield-enum-conversion
    -Wc++11-compat-pedantic
    -Wclass-varargs
    -Wconditional-uninitialized
    -Wthread-safety

    # Mistakes
    -Wconsumed
    -Wdirect-ivar-access
    -Wdisabled-macro-expansion
    -Wembedded-directive
    -Wexit-time-destructors
    -Wexpansion-to-defined
    -Wformat-pedantic
    -Widiomatic-parentheses
    -Winconsistent-missing-destructor-override
    -Winfinite-recursion
    -Wlocal-type-template-args
    -Wloop-analysis
    -Wmethod-signatures
    -Wmismatched-tags
    -Wmissing-braces
    -Wmissing-prototypes
    -Wmissing-variable-declarations
    -Wmost
    -Wmove
    -Wnonportable-system-include-path
    -Wnull-pointer-arithmetic
    -Wover-aligned
    -Woverriding-method-mismatch
    -Wpch-date-time
    -Wpragmas
    -Wreserved-id-macro
    -Wreserved-user-defined-literal
    -Wretained-language-linkage
    -Wsemicolon-before-method-body
    -Wsometimes-uninitialized
    -Wstring-conversion
    -Wsuper-class-method-mismatch
    -Wtautological-compare
    -Wundefined-reinterpret-cast
    -Wunreachable-code
    -Wunreachable-code-break
    -Wunreachable-code-loop-increment
    -Wunreachable-code-return
    -Wvector-conversion

    # Sanitizing
    -Wcomma
    -Wduplicate-enum
    -Wduplicate-method-arg
    -Wduplicate-method-match
    -Wdynamic-exception-spec
    -Wempty-translation-unit
    -Wexplicit-ownership-type
    -Wignored-qualifiers
    -Wimplicit
    -Wkeyword-macro
    -Wnewline-eof
    -Wredundant-parens
    -Wstatic-in-inline
    -Wstrict-prototypes
    -Wweak-template-vtables
    -Wweak-vtables
    -Wzero-length-array

    # Arrays
    -Warray-bounds-pointer-arithmetic
    -Wextended-offsetof
    -Wflexible-array-extensions

    # Arithmetics
    -Wfloat-conversion
    -Wfloat-overflow-conversion
    -Wfloat-zero-conversion
    -Wshorten-64-to-32
    -Wsign-compare
    -Wsign-conversion

    # Advices
    -Wcomment
    -Wdocumentation
    -Wdocumentation-pedantic
    -Wglobal-constructors
    -Wgnu
    -Wheader-hygiene
    -Wunneeded-internal-declaration
    -Wunneeded-member-function
    -Wvla
    -Wsuggest-final-types
    -Wsuggest-final-methods
    -Wsuggest-override
    -Wshift-overflow
    -Wshift-negative-value
    -Wnull-dereference
    -Wunused-macros

    # Clang >= 6.0.0
    -Wextra-semi
  )
else()
  # compile with other compiler
  set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
  set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

  add_compile_options(
    # CUDA compiler flags
    --forward-unknown-opts
    -Xptxas -O3,-v

    "$<gcc_like_cxx>:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused"
    "$<msvc_cxx:-W3>"
  )
endif()