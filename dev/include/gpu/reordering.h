#pragma once

#include "../common_concepts.hxx"
#include "./errors.h"
#include "./errors.h"
#include "./iterator.h"
#include "./iterator.h"
#include "./kernel_utils.h"
#include "./kernel_utils.h"
#include "./traits.hxx"
#include "./types.h"
#include "./types.h"
#include <concepts>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace device {
namespace reordering {

namespace kernel {

/// @brief swaps arr[idxFrom] -> arr[idxTo], assuming idxFrom/To is unique
/// (no race conditions). Indices must not exceed arr end.
/// @tparam T
/// @tparam Idx
/// @param arr
/// @param idxFrom
/// @param IdxTo
/// @param nIdx
template <std::copyable T, std::integral Idx>
__global__ void
swap_sparse(device_ptr_inout<T> arr, device_ptr_in<Idx> idxFrom,
            device_ptr_in<Idx> idxTo, const std::size_t nIdx) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nIdx)
        return;

    arr[idxTo[tid]] = arr[idxFrom[tid]];
}

template <typename T1, std::constructible_from<T1> T2,
          std::integral Idx = std::size_t>
__global__ void
select(device_ptr_in<T1> in, device_ptr_out<T2> out, device_ptr_in<Idx> indices,
       const unsigned long nIndices) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nIndices)
        return;

    out[tid] = in[indices[tid]];
}

} // namespace kernel

template <typename Iter, typename IterIdx>
inline void
swap_sparse_n(Iter begin, IterIdx idxFrom, IterIdx idxTo,
              const std::size_t nIdx, const cudaStream_t stream = 0) {

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(nIdx);

    if (nBlocks <= 1)
        return;

    using device::kernel::utils::iterToRawPtr;

    kernel::swap_sparse<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT, 0,
                          stream>>>(iterToRawPtr(begin), iterToRawPtr(idxFrom),
                                    iterToRawPtr(idxTo), nIdx);
    errors::check();
}

template <typename IterIn, typename IterOut, typename IterIdx>
    requires IterValConvertibleTo<IterIn, IterOut> and
             std::integral<typename IterIdx::value_type>
inline void
select_k(IterIn in, IterOut out, IterIdx indicesBegin, std::size_t k,
         const cudaStream_t stream = 0) {

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(k);

    using device::kernel::utils::iterToRawPtr;

    kernel::select<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT, 0,
                     stream>>>(iterToRawPtr(in), iterToRawPtr(out),
                               iterToRawPtr(indicesBegin), k);
}

} // namespace reordering
} // namespace device
