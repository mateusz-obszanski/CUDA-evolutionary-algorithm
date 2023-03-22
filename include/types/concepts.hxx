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

template <typename T, typename ToCompare>
concept GtComparableWith = requires(T x, ToCompare y) {{x > y} -> std::convertible_to<bool>; };

} // namespace concepts
} // namespace types
