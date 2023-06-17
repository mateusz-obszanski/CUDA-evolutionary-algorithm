#pragma once

#include "../common_concepts.hxx"

namespace device {
namespace functors {

template <typename V = float, typename Thresh = V>
    requires ComparableGT<V, Thresh>
class Threshold {
public:
    const Thresh thresh;

    __host__ __device__ __forceinline__ bool
    operator()(const float p) const noexcept { return p > thresh; }
};

template <typename V = float, typename Thresh = V>
    requires ComparableGT<V, Thresh>
class ThresholdLess {
public:
    const Thresh thresh;

    __host__ __device__ __forceinline__ bool
    operator()(const float p) const noexcept { return p < thresh; }
};

} // namespace functors
} // namespace device
