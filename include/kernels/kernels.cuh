#pragma once

#include "../concepts.hxx"
#include "../cuda_types.cuh"

namespace kernel {
    template <typename T>
    __global__ void
    fill(dRawVecOut<T> arr, const std::size_t nElems, const T fillval) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nElems)
            return;

        arr[idx] = fillval;
    }

    template <typename SrcT, concepts::ConstructibleButDifferent<SrcT> DestT>
    __global__ void
    copy(dRawVecOut<DestT> dest, dRawVecIn<SrcT> src, const std::size_t nElems) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nElems)
            return;

        dest[idx] = static_cast<DestT>(src[idx]);
    }
} // namespace kernel
