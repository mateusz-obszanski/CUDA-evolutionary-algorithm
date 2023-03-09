#pragma once

#include "../concepts.hxx"
#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../kernels/functional.cuh"
#include "../types.hxx"

namespace launcher {
    // TODO reduction
    // TODO mapReduce
    // TODO mapContext with or without padding and mode (e.g. for convolution)

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    map(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f) {
        constexpr std::size_t blockSize = 1024;
        const auto            nBlocks   = cuda_utils::divCeil<std::size_t>(len, blockSize);

        kernel::map<<<nBlocks, blockSize>>>(out, in, len, f);
        cuda_utils::host::checkKernelLaunch("map");
    }

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    map(
        dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f, culong begin) {

        map(out + begin, in + begin, len - begin, f);
    }

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    map(
        dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f, culong begin, culong end) {

        map(out + begin, in + begin, len - begin - end, f);
    }
} // namespace launcher
