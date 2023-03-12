#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../exceptions.hxx"
#include "../kernels/functional.cuh"
#include "../types.hxx"
#include "./_utils.cuh"

namespace launcher {
    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f) {
        const auto nBlocks = utils::calcBlockNum1D(len);

        kernel::transform<<<nBlocks, utils::BLOCK_SIZE_1D>>>(out, in, len, f);
        cuda_utils::host::checkKernelLaunch("transform");
    }

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(
        dRawVecOut<B> out,
        dRawVecIn<A>  in,
        culong        len,
        F             f,
        culong        begin) {

        transform(out + begin, in + begin, len - begin, f);
    }

    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(
        dRawVecOut<B> out,
        dRawVecIn<A>  in,
        culong        len,
        F             f,
        culong        begin,
        culong        end) {

        transform(out + begin, in + begin, len - begin - end, f);
    }

    using kernel::ReduceStrategy;
    using kernel::Reductible;

    template <
        Reductible                 T,
        Reductible                 Acc = T,
        concepts::Reductor<T, Acc> F   = cuda_ops::Add2<Acc, T>>
    inline void
    reduce(
        dRawVecOut<Acc> dBuff,
        dRawVecIn<T>    in,
        culong          len,
        F               f        = F(),
        ReduceStrategy  strategy = ReduceStrategy::RECURSE) {

        if (strategy != ReduceStrategy::RECURSE)
            throw exceptions::NotImplementedError(
                "launcher::reduce strategy != ReduceStrategy::RECURSE");

        const auto nBlocks = utils::calcBlockNum1D(len);

        kernel::reduce<T, Acc, F>
            <<<nBlocks, utils::BLOCK_SIZE_1D>>>(
                dBuff, in, len, f, strategy);

        cuda_utils::host::checkKernelLaunch("reduce");
    }

    template <
        Reductible                 T,
        Reductible                 Acc = T,
        concepts::Reductor<T, Acc> F   = cuda_ops::Add2<Acc, T>>
    inline Acc
    reduce(
        dRawVecIn<T>   in,
        culong         len,
        F              f        = F(),
        ReduceStrategy strategy = ReduceStrategy::RECURSE) {

        const auto nBlocks = utils::calcBlockNum1D(len);

        utils::SimpleDeviceBuffer<Acc> buffer(nBlocks);

        launcher::reduce<T, Acc, F>(buffer.data(), in, len, f, strategy);

        const auto hBuffer = buffer.toHost();

        if (strategy == ReduceStrategy::RECURSE)
            return hBuffer[0]; // already accumulated

        // TODO accumulate on CPU
        throw exceptions::NotImplementedError(
            "launcher::reduce strategy != ReduceStrategy::RECURSE");
    }
} // namespace launcher
