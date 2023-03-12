#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_types.cuh"
#include "../types.hxx"
#include <concepts>

namespace kernel {
    enum class GridDimensionality {
        NONE    = 0,
        D1      = 1,
        D2      = 2,
        D3      = 3,
        UNKNOWN = D3, // D3 will work with every dimension, so it is a safe fallback
    };

    namespace {
        constexpr __device__ auto
        calcOffsetX() {
            return blockIdx.x * blockDim.x + threadIdx.x;
        }

        constexpr __device__ auto
        calcOffsetY() {
            return blockIdx.y * blockDim.y + threadIdx.y;
        }

        constexpr __device__ auto
        calcOffsetZ() {
            return blockIdx.z * blockDim.z + threadIdx.z;
        }

        constexpr __device__ auto
        calcThreadNumAlongX() {
            return gridDim.x * blockDim.x;
        }

        constexpr __device__ auto
        calcThreadNumAlongY() {
            return gridDim.y * blockDim.y;
        }

        constexpr __device__ auto
        calcThreadNumXY() {
            return calcThreadNumAlongX() * calcThreadNumAlongY();
        }

        template <GridDimensionality>
        __device__ constexpr auto
        calcLinearizedIndex();

        template <>
        __device__ constexpr auto
        calcLinearizedIndex<GridDimensionality::D1>() {
            return calcOffsetX();
        }

        template <>
        __device__ constexpr auto
        calcLinearizedIndex<GridDimensionality::D2>() {
            return calcOffsetY() * calcThreadNumAlongX() + calcLinearizedIndex<GridDimensionality::D1>();
        }

        template <>
        __device__ constexpr auto
        calcLinearizedIndex<GridDimensionality::D3>() {
            return calcOffsetZ() * calcThreadNumXY() + calcLinearizedIndex<GridDimensionality::D2>();
        }
    } // namespace

    template <
        typename A,
        typename B = A,
        concepts::MappingFn<A, B> F,
        GridDimensionality        Dims = GridDimensionality::D1>
    /// @brief Dimensions:
    /// - grid: ceil(width / FUNCTIONAL_THREADS_IN_BLOCK) x 1 x 1
    /// - block: FUNCTIONAL_THREADS_IN_BLOCK x 1 x 1
    __global__ void
    transform(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f) {
        const auto idx = calcLinearizedIndex<Dims>();

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
    transform_if_not(dRawVecOut<B> out, dRawVecIn<A> in, culong len, F f, P p) {
        const auto idx = calcLinearizedIndex<Dims>();

        if (idx >= len)
            return;

        const auto a = in[idx];

        if (p(a))
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
        dRawVecOut<Acc> buffOut,
        dRawVecIn<T>    in,
        culong          len,
        F               f        = cuda_ops::Add2<Acc, T>(),
        ReduceStrategy  strategy = ReduceStrategy::RECURSE) {

        __shared__ Acc s_buff[REDUCE_BUFF_LENGTH];

        const auto idx = calcLinearizedIndex<Dims>();

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
