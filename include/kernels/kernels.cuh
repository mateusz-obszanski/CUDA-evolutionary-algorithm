#pragma once

#include "../cuda_types.cuh"

namespace kernel {
    template <typename T>
    __global__ void fill(dRawVecOut<T> arr, const T fillval, const std::size_t nElems) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nElems)
            return;

        arr[idx] = fillval;
    }
} // namespace kernel
