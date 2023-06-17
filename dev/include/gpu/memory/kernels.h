#pragma once

#include "../../common_concepts.hxx"
#include "../types.h"
#include "../types.h"
#include <cuda/std/cstddef>

namespace device {
namespace memory {
namespace kernel {

/// For the same SrcT and DstT use cudaMemcpy, not a kernel
template <typename SrcT, ConstructibleButDifferentFrom<SrcT> DstT>
    requires Mutable<DstT>
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
