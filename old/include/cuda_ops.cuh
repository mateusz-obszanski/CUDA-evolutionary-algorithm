#pragma once

#include "./concepts.hxx"

#define _CUDA_OPS_DEFINE_ARITHMETIC_BINOP(NAME, CONCEPT, EXPRESSION) \
    template <typename R, typename T = R, typename U = T>            \
        requires CONCEPT<R, T, U>                                    \
    class NAME {                                                     \
    public:                                                          \
        [[nodiscard]] constexpr __host__ __device__ R                \
        operator()(T x, U y) const {                                 \
            return EXPRESSION;                                       \
        }                                                            \
    };

namespace cuda_ops {
    _CUDA_OPS_DEFINE_ARITHMETIC_BINOP(Add2, concepts::Addable, x + y)
    _CUDA_OPS_DEFINE_ARITHMETIC_BINOP(Sub2, concepts::Subtractable, x - y)
    _CUDA_OPS_DEFINE_ARITHMETIC_BINOP(Mul2, concepts::Multiplicable, x* y)
    _CUDA_OPS_DEFINE_ARITHMETIC_BINOP(Div2, concepts::Divisible, x / y)

    // template <template <typename RR, typename XX, typename... YYs> typename
    // F, typename R, typename X = R, typename... Ys>
    // // requires concepts::Function<F<R, X, Ys...>, R, X, Ys...> ||
    // concepts::Functor<F<R, X, Ys...>, R, X, Ys...> struct Partial {
    //     // template <template <typename RR, typename XX, typename... YYs>
    //     typename F, typename R, typename X, typename... Ys>
    //     // struct Partial<F<R, X, Ys...>, R, X, Ys...> {
    // private:
    //     using _F = F<R, X, Ys...>;

    //     _F f;
    //     X  x;

    // public:
    //     Partial(_F f, X x) : f{f}, x{x} {}
    //     Partial(X x) : Partial(_F(), x) {}

    //     constexpr __host__ __device__ R
    //     operator()(Ys... ys) {
    //         return f(x, ys...);
    //     }
    // };

    // template <template <typename RR, typename XX, typename YY> typename F,
    // typename R, typename X, typename Y>
    // // requires concepts::Function<F<R, X, Ys...>, R, X, Ys...> ||
    // concepts::Functor<F<R, X, Ys...>, R, X, Ys...> struct Partial<F, R, X, Y>
    // {
    //     // template <template <typename RR, typename XX, typename... YYs>
    //     typename F, typename R, typename X, typename... Ys>
    //     // struct Partial<F<R, X, Ys...>, R, X, Ys...> {
    // private:
    //     using _F = F<R, X, Y>;

    //     _F f;
    //     X  x;

    // public:
    //     Partial(_F f, X x) : f{f}, x{x} {}
    //     Partial(X x) : Partial(_F(), x) {}

    //     constexpr __host__ __device__ R
    //     operator()(Y y) {
    //         return f(x, y);
    //     }
    // };

    template <
        typename F,
        typename R,
        typename X = R,
        typename Y = X,
        typename... Ys>
        requires concepts::Function<F, R, X, Y, Ys...>
    struct Partial {
    private:
        F f;
        X x;

    public:
        [[nodiscard]] Partial(F f, X x) : f(f), x(x) {}

        [[nodiscard]] Partial(X x) : Partial(F(), x) {}

        [[nodiscard]] constexpr __device__ R
        operator()(Y y, Ys... ys) {
            return f(x, y, ys...);
        }
    };

    template <typename F, typename R, typename X>
        requires concepts::Function<F, R, X>
    struct Partial<F, R, X, void> {
    private:
        F f;
        X x;

    public:
        [[nodiscard]] Partial(F f, X x) : f(f), x(x) {}
        [[nodiscard]] Partial(X x) : Partial(F(), x){};

        [[nodiscard]] constexpr __device__ R
        operator()() {
            return f(x);
        }
    };

    template <
        typename F1,
        typename F2,
        typename T1,
        typename R1 = T1,
        typename T2 = T1,
        typename R2 = T2>
        requires concepts::MappingFn<F1, T1, R1> and
                 concepts::MappingFn<F2, T2, R2> and std::convertible_to<R2, T1>
    struct Compose2 {
    private:
        F1 f1;
        F2 f2;

    public:
        [[nodiscard]] Compose2(F1 f1, F2 f2) : f1{f1}, f2{f2} {}

        [[nodiscard]] constexpr __host__ __device__ R1
        operator()(T2 x) {
            return f1(f2(x));
        }
    };

    template <
        typename F1,
        typename F2,
        typename T1,
        typename R1 = T1,
        typename T2 = R1,
        typename R2 = T2>
        requires concepts::MappingFn<F1, T1, R1> and
                 concepts::MappingFn<F2, T2, R2> and std::convertible_to<R1, T2>
    struct Pipe2 {
    private:
        Compose2<F2, F1, T2, R2, T1, R1> composed;

    public:
        [[nodiscard]] Pipe2(const F1 f1, const F2 f2) : composed{f2, f1} {}

        [[nodiscard]] constexpr __host__ __device__ R2
        operator()(T1 x) {
            return composed(x);
        }
    };

    template <typename P, typename... Ts>
        requires concepts::Predicate<P, Ts...>
    struct Negate {
    private:
        P p;

    public:
        [[nodiscard]] constexpr __host__ __device__ bool
        operator()(Ts... xs...) {
            return !static_cast<bool>(p(xs...));
        }
    };
} // namespace cuda_ops
