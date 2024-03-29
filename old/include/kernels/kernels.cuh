#pragma once

#include "../concepts.hxx"
#include "../cuda_types.cuh"

namespace kernel {
    template <typename T>
    __global__ void
    fill(device_ptr_out<T> arr, const std::size_t nElems, const T fillval) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nElems)
            return;

        arr[idx] = fillval;
    }

    template <typename SrcT, concepts::ConstructibleButDifferent<SrcT> DestT>
    __global__ void
    copy(
        device_ptr_out<DestT> dest,
        device_ptr_in<SrcT>   src,
        const std::size_t     nElems) {

        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nElems)
            return;

        dest[idx] = static_cast<DestT>(src[idx]);
    }

    template <typename T>
    concept CountingAble = concepts::Addable<T> && concepts::Multiplicable<T> &&
                           std::constructible_from<T, unsigned int>;

    template <CountingAble T>
    inline void
    counting(
        device_ptr_out<T> out,
        const std::size_t nElems,
        const T           start,
        const T           step) {

        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nElems)
            return;

        const auto n = start + static_cast<T>(idx) * step;

        out[idx] = n;
    }
} // namespace kernel
