#pragma once

#include <cuda_runtime.h>

namespace device {
    namespace common {
        enum class GridDimensionality {
            NONE    = 0,
            D1      = 1,
            D2      = 2,
            D3      = 3,
            UNKNOWN = D3, // D3 will work with every dimension, so it is a safe fallback
        };

        template <GridDimensionality Dims>
        class Dimensionable {
        public:
            static inline constexpr auto dims = Dims;
        };

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
    } // namespace common
} // namespace device
