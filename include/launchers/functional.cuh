#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../kernels/functional.cuh"
#include "../types.hxx"
#include "./_utils.cuh"

namespace launcher {
    // TODO reduction
    // TODO mapReduce
    // TODO mapContext with or without padding and mode (e.g. for convolution)

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f) {
        const auto&& [nBlocks, blockSize] = utils::calcLaunchParams1D(len);

        kernel::transform<<<nBlocks, blockSize>>>(out, in, len, f);
        cuda_utils::host::checkKernelLaunch("transform");
    }

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(
        dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f, culong begin) {

        transform(out + begin, in + begin, len - begin, f);
    }

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(
        dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f, culong begin, culong end) {

        transform(out + begin, in + begin, len - begin - end, f);
    }

    using kernel::ReduceStrategy;
    using kernel::Reductible;

    template <Reductible T, Reductible Acc = T, concepts::Reductor<T, Acc> F = cuda_ops::Add<Acc, T>>
    inline Acc
    reduce(dRawVecOut<Acc> dBuff, dRawVecIn<Acc> in, culong len, F f, ReduceStrategy strategy = ReduceStrategy::RECURSE) {
        const auto&& [nBlocks, blockSize] = utils::calcLaunchParams1D(len);

        kernel::reduce<<<nBlocks, blockSize>>>(dBuff, in, len, f, strategy);
        cuda_utils::host::checkKernelLaunch("reduce");
    }

} // namespace launcher
