#pragma once

#include <concepts>
#include <iterator>
#include <string>

#define _CONCEPTS_DEFINE_BINOP_CONCEPT(NAME, EXPRESSION)         \
    template <typename R, typename T = R, typename U = T>        \
    concept NAME = requires(T x, U y) {                          \
                       { EXPRESSION } -> std::convertible_to<R>; \
                   };

namespace concepts {
    _CONCEPTS_DEFINE_BINOP_CONCEPT(Addable, x + y)
    _CONCEPTS_DEFINE_BINOP_CONCEPT(Subtractable, x - y)
    _CONCEPTS_DEFINE_BINOP_CONCEPT(Multiplicable, x* y)
    _CONCEPTS_DEFINE_BINOP_CONCEPT(Divisible, x / y)

    template <typename T, typename U = T>
    concept WeaklyComparable = requires(T x, U y) {
                                   { x < y } -> std::same_as<bool>;
                               };

    template <typename F, typename R, typename... Ts>
    concept Function = std::is_invocable_r_v<R, F, Ts...>;

    template <typename F, typename R, typename... Ts>
    concept Functor = std::is_object_v<F> and std::regular_invocable<F, Ts...>;

    template <typename F, typename X, typename Y = X>
    concept MappingFn = Function<F, Y, X>;

    template <typename F, typename X>
    concept Endomorphism = MappingFn<F, X, X>;

    template <typename T>
    concept Stringifiable = std::convertible_to<T, std::string>;

    template <typename _InIt, typename T>
    concept InputIter =
        std::input_iterator<_InIt> and
        requires(_InIt it) {
            { *it } -> std::convertible_to<T>;
        };

    template <typename DestT, typename SrcT>
    concept ConstructibleButDifferent =
        (std::constructible_from<DestT, SrcT> and
         not std::same_as<DestT, SrcT>);

    template <typename F, typename R, typename T1, typename T2>
    concept BinOp = std::is_invocable_r_v<R, F, T1, T2>;

    template <typename F, typename T, typename Acc = T>
    concept Reductor = BinOp<F, Acc, T, Acc>;

    template <typename F, typename... Ts>
    concept Predicate = Function<F, bool, Ts...>;
} // namespace concepts
