#pragma once

#include <concepts>
#include <iterator>
#include <string>

namespace concepts {
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
} // namespace concepts
