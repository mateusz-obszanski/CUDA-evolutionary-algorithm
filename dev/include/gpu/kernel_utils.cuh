#pragma once

#include "../common_concepts.hxx"
#include <concepts>
#include <thrust/device_ptr.h>

namespace device {
namespace kernel {
namespace utils {

constexpr unsigned int WARP_SIZE          = 32;
constexpr unsigned int BLOCK_SIZE_DEFAULT = WARP_SIZE;

template <std::convertible_to<double> T = int>
constexpr __host__ __device__ T
divCeil(T x, T y) {
    return ceil(static_cast<double>(x) / y);
}

__host__ __device__ constexpr unsigned int 
calcBlockNum1D(const unsigned int  nElems, const unsigned int  blockSize = BLOCK_SIZE_DEFAULT) {
    return divCeil(nElems, blockSize);
}

template <typename Iter>
inline auto
iterToRawPtr(const Iter iter) noexcept {
    return thrust::raw_pointer_cast(&(*iter));
}

template <typename X, typename Y>
    requires MutuallyConvertible<X, Y>
__host__ __device__ inline void
swap(const X& x, const Y& y) {
    const auto temp = y;

    x = y;
    y = temp;
}

template <typename X, typename Y>
    requires MutuallyConvertible<X, Y>
__host__ __device__ inline void
swap_at(X* const x, Y* const y, const std::size_t i) {
    swap(x[i], y[i]);
}

template <typename X, typename Y>
    requires MutuallyConvertible<X, Y>
__host__ __device__ inline void
swap_at_if(X* const xs, Y* const ys, const std::size_t i, const bool pred) {
    if (pred)
        swap_at(xs, ys, i);
}

template <typename X, typename Y>
    requires MutuallyConvertible<X, Y>
__host__ __device__ inline void
swap_at_if(X* const xs, Y* const ys, const std::size_t i, bool* const preds) {
    swap_at_if(xs, ys, i, preds[i]);
}

} // namespace utils
} // namespace kernel
} // namespace device
