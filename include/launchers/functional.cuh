#pragma once

#include "../concepts.hxx"
#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../kernels/functional.cuh"
#include "../types.hxx"

namespace launcher {
    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void map(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f) {
        constexpr std::size_t blockSize = 1024;
        const auto            nBlocks   = cuda_utils::divCeil<std::size_t>(len, blockSize);

        kernel::map<<<nBlocks, blockSize>>>(out, in, len, f);
        cuda_utils::host::checkKernelLaunch("map");
    }
} // namespace launcher