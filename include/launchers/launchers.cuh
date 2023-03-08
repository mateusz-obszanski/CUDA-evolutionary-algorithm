#pragma once

#include "../cuda_utils/cuda_utils.cuh"
#include "../kernels/kernels.cuh"
#include <cstddef>

namespace launcher {
    template <typename T>
    void fill(T* const arr, const T fillval, const std::size_t nElems) {
        constexpr std::size_t blockSize = 1024;
        const auto            nBlocks   = cuda_utils::divCeil<std::size_t>(nElems, blockSize);

        kernel::fill<<<nBlocks, blockSize>>>(arr, fillval, nElems);
        cuda_utils::host::checkKernelLaunch("fill");
    }
} // namespace launcher
