#pragma once

#include "../types/concepts.hxx"

namespace device {
namespace functors {

template <typename V = float, typename Thresh = V>
    requires types::concepts::GtComparableWith<V, Thresh>
class Threshold {
public:
    const Thresh thresh;

    __host__ __device__ bool
    operator()(const float p) const noexcept { return p > thresh; }
};

template <typename V = float, typename Thresh = V>
    requires types::concepts::GtComparableWith<V, Thresh>
class ThresholdLess {
public:
    const Thresh thresh;

    __host__ __device__ bool
    operator()(const float p) const noexcept { return p < thresh; }
};

} // namespace functors
} // namespace device
