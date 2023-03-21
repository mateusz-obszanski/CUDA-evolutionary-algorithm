#pragma once

#include <concepts>
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
concept Stringifiable = std::constructible_from<std::string, T>;

template <typename T>
concept NotStringifiable = (not Stringifiable<T>);

template <typename T>
concept Const = std::is_const_v<T>;

template <typename T>
concept Mutable = (not Const<T>);

} // namespace concepts
} // namespace types
