#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_types.cuh"
#include "../device_common.cuh"
#include "../types.hxx"
#include <concepts>

namespace kernel {
    using typename device::common::GridDimensionality;

    template <
        typename A,
        typename B = A,
        concepts::MappingFn<A, B> F,
        GridDimensionality        Dims = GridDimensionality::D1>
    /// @brief Dimensions:
    /// - grid: ceil(width / FUNCTIONAL_THREADS_IN_BLOCK) x 1 x 1
    /// - block: FUNCTIONAL_THREADS_IN_BLOCK x 1 x 1
    __global__ void
    transform(device_ptr_out<B> out, device_ptr_in<A> in, culong len, F f) {
        const auto idx = device::common::calcLinearizedIndex<Dims>();

        if (idx >= len)
            return;

        out[idx] = f(in[idx]);
    }

    template <
        typename A,
        typename B = A,
        concepts::MappingFn<A, B> F,
        concepts::Predicate<A>    P,
        GridDimensionality        Dims = GridDimensionality::D1>
    __global__ void
    transform_if_not(device_ptr_out<B> out, device_ptr_in<A> in, culong len, F f, P p) {
        const auto idx = device::common::calcLinearizedIndex<Dims>();

        if (idx >= len)
            return;

        const auto a = in[idx];

        if constexpr (p(a))
            return;

        out[idx] = f(a);
    }

    enum class ReduceStrategy {
        NORMAL,
        RECURSE
    };

    namespace {
        template <GridDimensionality>
        __device__ constexpr auto
        calcReductionWriteIndex();

        template <>
        __device__ constexpr auto
        calcReductionWriteIndex<GridDimensionality::D1>() {
            return blockIdx.x;
        };

        template <>
        __device__ constexpr auto
        calcReductionWriteIndex<GridDimensionality::D2>() {
            return blockIdx.y * gridDim.x + blockIdx.x;
        };

        template <>
        __device__ constexpr auto
        calcReductionWriteIndex<GridDimensionality::D3>() {
            return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        };
    } // namespace

    constexpr ulong REDUCE_BUFF_LENGTH = 1024;
    constexpr auto  REDUCE_THREAD_N    = REDUCE_BUFF_LENGTH;

    template <typename T>
    concept Reductible = std::is_default_constructible_v<T>;

    /// @brief Reduction along `x` (most nested) dimension
    /// @param buffOut length: gridDim.z * gridDim.y * (gridDim.x === no. of
    /// blocks)
    template <
        Reductible                 T,
        Reductible                 Acc  = T,
        concepts::Reductor<T, Acc> F    = cuda_ops::Add2<Acc, T>,
        GridDimensionality         Dims = GridDimensionality::D1>
    __global__ void
    reduce(
        device_ptr_out<Acc> buffOut,
        device_ptr_in<T>    in,
        culong              len,
        F                   f        = cuda_ops::Add2<Acc, T>(),
        ReduceStrategy      strategy = ReduceStrategy::RECURSE) {

        __shared__ Acc s_buff[REDUCE_BUFF_LENGTH];

        const auto idx = device::common::calcLinearizedIndex<Dims>();

        // load to shared buffer
        // treat default T value as 0
        s_buff[threadIdx.x] = idx < len ? in[idx] : T();

        // reduce shared buffer
        for (
            ulong nActiveThreads{blockDim.x >> 1};
            nActiveThreads > 0;
            nActiveThreads >>= 1) {

            __syncthreads();

            if (threadIdx.x >= nActiveThreads)
                break;

            s_buff[threadIdx.x] += s_buff[threadIdx.x + nActiveThreads];
        }

        // write to global buffer
        // global buffer might have multiple values and require
        if (threadIdx.x == 0) {
            buffOut[calcReductionWriteIndex<Dims>()] = s_buff[0];

            if (strategy == ReduceStrategy::NORMAL)
                return;

            if (blockIdx.x != 0)
                return;

            // recursive strategy, launch child grid - only from the globally
            // first thread

            // recursion requires CUDA separate compilation (cmake target
            // property CUDA_SEPARABLE_COMPILATION ON)

            // after one pass, `gridDim.x` values along X axis are left to
            // reduce (1 from each block along X axis)
            __syncthreads();

            reduce<T, Acc, F, Dims>
                <<<gridDim.x, REDUCE_THREAD_N>>>(
                    buffOut, buffOut, gridDim.x, f, strategy);
        }
    }
} // namespace kernel
