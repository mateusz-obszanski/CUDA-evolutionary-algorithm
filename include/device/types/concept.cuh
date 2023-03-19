#pragma once

#include "./types.cuh"
#include <concepts>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

namespace device {
namespace types {
namespace concepts {

template <typename T, typename Member>
concept HasMemberType = true;

template <typename A, typename T>
concept DeviceAllocator =
    requires(A a, cuda::std::size_t nBytes, device_ptr<T> p) {
        { a.allocate(nBytes) } -> std::same_as<device_ptr<T>>;
        { a.deallocate(p, nBytes) } -> std::same_as<void>;
    } and
    HasMemberType<A, typename A::value_type> and
    HasMemberType<A, typename A::pointer> and
    HasMemberType<A, typename A::size_type> and
    HasMemberType<A, typename A::difference_type> and
    HasMemberType<A, typename A::is_always_equal>;

} // namespace concepts
} // namespace types
} // namespace device
