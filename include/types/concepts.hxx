#pragma once

#include <concepts>

namespace types {
namespace concepts {

template <typename DestT, typename SrcT>
concept ConstructibleButDifferent =
    std::constructible_from<DestT, SrcT> and not
std::same_as<DestT, SrcT>;

} // namespace concepts
} // namespace types
