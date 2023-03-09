#pragma once

#include <concepts>
#include <iterator>
#include <string>

namespace concepts {
    template <typename T, typename U = T, typename R = T>
    concept Addable = requires(T x, T y) {{x + y} -> std::convertible_to<R>; };

    template <typename F, typename X, typename Y = X>
    concept MappingFn = std::is_invocable_r<Y, F, X>::value;

    template <typename F, typename X>
    concept Endomorphism = MappingFn<F, X, X>;

    template <typename T>
    concept Stringifiable = std::convertible_to<T, std::string>;

    template <typename _InIt, typename T>
    concept InputIter = std::input_iterator<_InIt> && requires(_InIt it) {{*it} -> std::convertible_to<T>; };

    template <typename DestT, typename SrcT>
    concept ConstructibleButDifferent = (std::constructible_from<DestT, SrcT> and not std::same_as<DestT, SrcT>);

    template <typename F, typename R, typename T1, typename T2>
    concept BinOp = std::is_invocable_r_v<R, F, T1, T2>;

    template <typename F, typename T, typename Acc = T>
    concept Reductor = BinOp<F, Acc, T, Acc>;
} // namespace concepts
