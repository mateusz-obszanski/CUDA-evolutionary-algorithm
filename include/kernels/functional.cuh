#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_types.cuh"
#include "../types.hxx"
#include <concepts>

namespace kernel {
    template <typename A, typename B, concepts::MappingFn<A, B> F>
    /// @brief Dimensions:
    /// - grid: ceil(width / FUNCTIONAL_THREADS_IN_BLOCK) x 1 x 1
    /// - block: FUNCTIONAL_THREADS_IN_BLOCK x 1 x 1
    __global__ void
    transform(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= len)
            return;

        out[idx] = f(in[idx]);
    }

    enum class ReduceStrategy { NORMAL,
                                RECURSE };

    constexpr ulong REDUCE_BUFF_LENGTH = 1024;
    constexpr auto  REDUCE_THREAD_N    = REDUCE_BUFF_LENGTH;

    template <typename T>
    concept Reductible = std::is_default_constructible_v<T>;

    /// @brief
    /// @param buffOut length == gridDim.x (no. of blocks)
    template <Reductible T, Reductible Acc = T, concepts::Reductor<T, Acc> F = cuda_ops::Add<Acc, T>>
    __global__ void
    reduce(dRawVecOut<Acc> buffOut, dRawVecIn<T> in, culong len, F f = cuda_ops::Add<Acc, T>{}, ReduceStrategy strategy = ReduceStrategy::RECURSE) {
        __shared__ Acc s_buff[REDUCE_BUFF_LENGTH];

        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        // load to shared buffer
        T zero();
        s_buff[threadIdx.x] = idx < len ? in[idx] : T();

        // reduce shared buffer
        for (ulong nActiveThreads{blockDim.x >> 1}; nActiveThreads > 0; nActiveThreads >>= 1) {
            __syncthreads();

            if (threadIdx.x >= nActiveThreads)
                break;

            s_buff[threadIdx.x] += s_buff[threadIdx.x + nActiveThreads];
        }

        // write to global buffer
        // global buffer might have multiple values and require
        if (threadIdx.x == 0) {
            buffOut[blockIdx.x] = s_buff[0];

            if (strategy == ReduceStrategy::NORMAL)
                return;

            if (blockIdx.x != 0)
                return;

            // recursive strategy, launch child grid - only from the globally first thread
            __syncthreads();
            // if gridDim.x cannot be passed to child grid (because it is local),
            // add kernel overload or something that will deduce len from its
            // grid dimension. Otherwise, just write launcher that runs
            // the kernel multiple times (if RECURSE strategy is given) and
            // reuses allocated memory. Otherwise, R.I.P.
            kernel::reduce<<<gridDim.x, REDUCE_THREAD_N>>>(buffOut, buffOut, gridDim.x, f, strategy);
        }
    }
} // namespace kernel
