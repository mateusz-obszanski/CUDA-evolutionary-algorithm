#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../exceptions.hxx"
#include "../kernels/functional.cuh"
#include "../types.hxx"
#include "./_utils.cuh"
#include <numeric>

namespace launcher {
    template <typename A, typename B, concepts::MappingFn<A, B> F>
    inline void
    transform(
        device_ptr_out<B> out,
        device_ptr_in<A>  in,
        culong            len,
        F                 f) {

        const auto nBlocks = utils::calcBlockNum1D(len);

        kernel::transform<<<nBlocks, utils::BLOCK_SIZE_1D>>>(out, in, len, f);

        cuda_utils::host::checkKernelLaunch("transform");
    }

    using kernel::ReduceStrategy;
    using kernel::Reductible;

    template <
        Reductible                 T,
        Reductible                 Acc = T,
        concepts::Reductor<T, Acc> F   = cuda_ops::Add2<Acc, T>>
    inline void
    reduce(
        device_ptr_out<Acc> dBuff,
        device_ptr_in<T>    in,
        culong              len,
        F                   f        = F(),
        ReduceStrategy      strategy = ReduceStrategy::RECURSE) {

        const auto nBlocks = utils::calcBlockNum1D(len);

        // if this results in error, check whether `dBuff` has enough memory
        // allocated for `nBlocks` `Acc` instances
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
        device_ptr_in<T> in,
        culong           len,
        F                f        = F(),
        ReduceStrategy   strategy = ReduceStrategy::RECURSE) {

        const auto nBlocks = utils::calcBlockNum1D(len);

        utils::SimpleDeviceBuffer<Acc> buffer(
            strategy == ReduceStrategy::RECURSE ? 1 : nBlocks);

        launcher::reduce<T, Acc, F>(buffer.data(), in, len, f, strategy);

        const auto hBuffer = buffer.toHost();

        if (strategy == ReduceStrategy::RECURSE)
            return hBuffer[0]; // already accumulated

        return std::accumulate(hBuffer.cbegin(), hBuffer.cend(), Acc());
    }
} // namespace launcher
