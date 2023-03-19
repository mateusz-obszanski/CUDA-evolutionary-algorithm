#pragma once

#include <concepts>

namespace types {
namespace concepts {

template <typename T, typename U>
concept DifferentFrom = (not std::same_as<T, U>);

template <typename T, typename U>
concept ConstructibleButDifferentFrom =
    std::constructible_from<T, U> and DifferentFrom<T, U>;

} // namespace concepts
} // namespace types
