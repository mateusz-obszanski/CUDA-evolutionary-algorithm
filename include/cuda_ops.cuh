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

    template <typename T, typename U = T, typename R = T>
        requires concepts::Multiplicable<T, U, R>
    struct Mult {
        __device__ R
        operator()(T x, U y) const {
            return x * y;
        }
    };

    template <typename F, typename R, typename T1 = R, typename T2 = R, typename... Ts>
        requires concepts::Fun<F, R, T1, T2, Ts...>
    struct Partial {
    private:
        F  f;
        T1 x;

    public:
        Partial(F f, T1 x) : f{f}, x{x} {}
        Partial(T1 x) : f{}, x{x} {}

        __device__ R
        operator()(T2 y, Ts... ys) const {
            return f(x, y, ys...);
        }
    };
} // namespace cuda_ops
