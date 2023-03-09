#pragma once

#include <cmath>
#include <concepts>

namespace cuda_utils {
    template <std::convertible_to<double> T = long long>
    inline constexpr __host__ __device__ T
    divCeil(T x, T y) {
        return ceil(static_cast<double>(x) / y);
    }
} // namespace cuda_utils
