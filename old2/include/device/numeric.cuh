#pragma once

#include "../types/concepts.hxx"
#include "./functors.cuh"
#include <concepts>

namespace device {
namespace numeric {

template <typename IterIn, typename IterOut, typename ThreshT>
    requires std::constructible_from<typename IterOut::value_type, bool> and
             types::concepts::GtComparableWith<typename IterIn::value_type, ThreshT>
inline void
threshold(IterIn begin, IterIn end, IterOut out, const ThreshT thresh) {
    using ValueT = typename IterIn::value_type;
    thrust::transform(
        begin, end, out, functors::Threshold<ValueT, ThreshT>{thresh});
}

template <typename IterIn, typename IterOut, typename ThreshT>
    requires std::constructible_from<typename IterOut::value_type, bool> and
             types::concepts::GtComparableWith<typename IterIn::value_type, ThreshT>
inline void
threshold_less(IterIn begin, IterIn end, IterOut out, const ThreshT thresh) {
    using ValueT = typename IterIn::value_type;
    thrust::transform(
        begin, end, out, functors::ThresholdLess<ValueT, ThreshT>{thresh});
}

} // namespace numeric
} // namespace device
