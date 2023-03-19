#pragma once

#include <concepts>

namespace device {
namespace kernel {
namespace utils {

constexpr uint WARP_SIZE          = 32;
constexpr uint BLOCK_SIZE_DEFAULT = WARP_SIZE;

template <std::convertible_to<double> T = int>
constexpr __host__ __device__ T
divCeil(T x, T y) {
    return ceil(static_cast<double>(x) / y);
}

__host__ __device__ constexpr uint
calcBlockNum1D(const uint nElems, const uint blockSize = BLOCK_SIZE_DEFAULT) {
    return divCeil(nElems, blockSize);
}

} // namespace utils
} // namespace kernel
} // namespace device
