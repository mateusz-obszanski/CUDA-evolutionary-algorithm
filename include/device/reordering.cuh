#pragma once

#include "../types/traits.hxx"
#include "./errors.cuh"
#include "./iterator.cuh"
#include "./kernel_utils.cuh"
#include "./types/types.cuh"
#include <concepts>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace device {
namespace reordering {

namespace concepts {

template <typename IterIn, typename IterOut, typename IterMask>
concept CopyMaskedAble = std::convertible_to<typename IterIn::value_type, typename IterOut::value_type> and
                         std::convertible_to<typename IterMask::value_type, bool>;

} // namespace concepts

namespace kernel {

/// @brief swaps arr[idxFrom] -> arr[idxTo], assuming idxFrom/To is unique
/// (no race conditions). Indices must not exceed arr end.
/// @tparam T
/// @tparam Idx
/// @param arr
/// @param idxFrom
/// @param IdxTo
/// @param nIdx
template <typename T, typename Idx>
__global__ void
swap_sparse(
    device_ptr_inout<T> arr,
    device_ptr_in<Idx> idxFrom, device_ptr_in<Idx> idxTo, const std::size_t nIdx) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nIdx)
        return;

    arr[idxTo[tid]] = arr[idxFrom[tid]];
}

} // namespace kernel

template <typename Iter, typename IterIdx>
inline void
swap_sparse_n(
    Iter begin, IterIdx idxFrom, IterIdx idxTo,
    const std::size_t nIdx, const cudaStream_t stream = 0) {

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(nIdx);

    kernel::swap_sparse<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT, 0, stream>>>(
        thrust::raw_pointer_cast(&begin[0]),
        thrust::raw_pointer_cast(&idxFrom[0]),
        thrust::raw_pointer_cast(&idxTo[0]),
        nIdx);
    errors::check();
}

} // namespace reordering
} // namespace device
