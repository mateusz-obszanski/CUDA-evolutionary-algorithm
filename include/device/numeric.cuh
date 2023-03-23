#pragma once

#include "./types/concepts.hxx"
#include <concepts>

namespace device {
namespace numeric {

template <typename IterIn, typename IterOut, typename ThreshT>
    requires std::constructible_from<typename IterOut::value_type, bool> and
             types::concepts::GtComparableWith<typename IterIn::value_type, ThreshT>
inline void
threshold(IterIn begin, IterIn end, IterOut out, const ThreshT thresh) {
    thrust::transform(
        begin, end, out, [=] __device__(const float p) -> bool { return p > thresh; });
}

} // namespace numeric
} // namespace device
