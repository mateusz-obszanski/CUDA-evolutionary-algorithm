#pragma once

#include "../../types/concepts.hxx"
#include "../types/types.cuh"
#include <cuda/std/cstddef>

namespace device {
namespace memory {
namespace kernel {

/// For the same SrcT and DstT use cudaMemcpy, not a kernel
template <typename SrcT, ::types::concepts::ConstructibleButDifferentFrom<SrcT> DstT>
__global__ void
copy(
    device_ptr_out<DstT>    dst,
    device_ptr_in<SrcT>     src,
    const cuda::std::size_t nElems) {

    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nElems)
        return;

    dst[idx] = static_cast<DstT>(src[idx]);
}

} // namespace kernel
} // namespace memory
} // namespace device
