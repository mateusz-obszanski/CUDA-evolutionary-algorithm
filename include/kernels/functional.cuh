#pragma once

#include "../concepts.hxx"
#include "../cuda_types.cuh"
#include "../types.hxx"

namespace kernel {
    constexpr uint WARP                        = 32;
    constexpr uint FUNCTIONAL_THREADS_IN_BLOCK = 1024;

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    /// @brief Dimensions:
    /// - grid: ceil(width / FUNCTIONAL_THREADS_IN_BLOCK) x 1 x 1
    /// - block: FUNCTIONAL_THREADS_IN_BLOCK x 1 x 1
    inline __global__ void map(dRawVecOut<T> out, dRawVecIn<T> in, culong len, F f) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= len)
            return;

        out[idx] = f(in[idx]);
    }
} // namespace kernel
