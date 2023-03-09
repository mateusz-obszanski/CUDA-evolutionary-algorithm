#pragma once

#include "./concepts.hxx"

namespace cuda_ops {
    template <typename T, typename U = T, typename R = T>
        requires concepts::Addable<T, U, R>
    struct Add {
        __device__ R
        operator()(T x, U y) const {
            return x + y;
        }
    };
} // namespace cuda_ops
