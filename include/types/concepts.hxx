#pragma once

#include <concepts>
#include <sstream>
#include <string>
#include <type_traits>

namespace types {
namespace concepts {

template <typename T>
concept Number = std::integral<T> or std::floating_point<T>;

template <typename T, typename... U>
concept AnyOf = (std::same_as<T, U> || ...);

template <typename T, typename... U>
concept DifferentFrom = (not std::same_as<T, U> || ...);

template <typename T, typename U>
concept ConstructibleButDifferentFrom =
    std::constructible_from<T, U> and DifferentFrom<T, U>;

template <typename T>
concept StrStreamStringifiable = requires(T t) { std::stringstream{} << t; };

template <typename T>
concept NotStrStreamStringifiable = (not StrStreamStringifiable<T>);

template <typename T>
concept Const = std::is_const_v<T>;

template <typename T>
concept Mutable = (not Const<T>);

template <typename T, typename U>
concept EqByteSize = sizeof(T) == sizeof(U);

template <typename T, typename U>
concept NeqByteSize = (not EqByteSize<T, U>);

template <typename T, typename ToCompare>
concept GtComparableWith =
    requires(T x, ToCompare y) {{x > y} -> std::convertible_to<bool>; };

template <typename IterL, typename IterR>
concept ConvertibleIterVal =
    std::convertible_to<typename IterL::value_type, typename IterR::value_type>;

template <typename Iter, typename T>
concept IterValConvertibleTo =
    std::convertible_to<typename Iter::value_type, T>;

} // namespace concepts
} // namespace types
