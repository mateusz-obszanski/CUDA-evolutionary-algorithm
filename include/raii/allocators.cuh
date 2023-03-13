#pragma once

#include "../cuda_utils/exceptions.cuh"
#include "../device_common.cuh"
#include "../macros.hxx"
#include <concepts>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>

namespace raii {
    namespace allocator {
        using device::common::GridDimensionality;

        DEFINE_CUDA_ERROR(DeviceBadAllocError, "Could not allocate memory on GPU")
        DEFINE_SIMPLE_ERROR(DeviceBadArrayNewLength, "Too big device array length")

        template <typename T>
        class BaseDeviceAllocator {
        public:
            using value_type      = T;
            using pointer         = T*;
            using size_type       = std::size_t;
            using is_always_equal = std::true_type; // stateless allocator
        };

        namespace {
            using enum GridDimensionality;
        }

        template <typename T, GridDimensionality = D1>
        class DeviceAllocator;

        template <typename T>
        using DeviceAllocatorD1 = DeviceAllocator<T, D1>;

        template <typename T>
        using DeviceAllocatorD2 = DeviceAllocator<T, D2>;

        template <typename T>
        using DeviceAllocatorD3 = DeviceAllocator<T, D3>;

        template <typename T>
        class DeviceAllocator<T, D1> : public BaseDeviceAllocator<T> {
        public:
            using cls = DeviceAllocatorD1<T>;

            [[nodiscard]] DeviceAllocator() = default;

            template <typename U>
            [[nodiscard]] inline DeviceAllocator(
                const DeviceAllocator<U, D1>&) noexcept {}

            [[nodiscard("DEVICE MEMORY LEAK")]] cls::pointer
            allocate(cls::size_type n);

            void
            deallocate(cls::pointer p, cls::size_type n);

            void
            deallocate(cls::pointer p);
        };

        template <
            template <typename EE, typename... Params>
            typename Allocator,

            typename ElemT,
            GridDimensionality Dims,
            typename... P>
        concept IsAllocatorTemplate = std::same_as<Allocator<ElemT, P...>, DeviceAllocator<ElemT, Dims>>;

        template <typename T>
        inline DeviceAllocatorD1<T>::pointer
        DeviceAllocatorD1<T>::allocate(
            typename DeviceAllocatorD1<T>::size_type n) {

            const auto maxElemN =
                std::numeric_limits<typename cls::size_type>::max() / sizeof(T);

            if (n > maxElemN)
                throw DeviceBadArrayNewLength();

            typename cls::pointer p;
            const auto            status = cudaMalloc<T>(&p, n);

            DeviceBadAllocError::check(status);

            return p;
        }

        DEFINE_CUDA_ERROR(DeviceFreeError, "Could not free device memory")

        template <typename T>
        inline void
        DeviceAllocatorD1<T>::deallocate(
            typename cls::pointer p,
            typename cls::size_type) {

            const auto status = cudaFree(p);
            DeviceFreeError::check(status);
        }

        template <typename T>
        inline void
        DeviceAllocatorD1<T>::deallocate(typename cls::pointer p) {
            deallocate(p, 0);
        }
    } // namespace allocator
} // namespace raii
