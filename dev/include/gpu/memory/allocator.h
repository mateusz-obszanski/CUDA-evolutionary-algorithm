#pragma once

#include "../../common_concepts.hxx"
#include "../errors.h"
#include "../types.h"
#include <cuda/std/cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace device {
namespace memory {
namespace allocator {

enum class AllocMode {
    SYNC,
    ASYNC,
    MANAGED,
    HOST_PINNED,
};

namespace {

using size_t = cuda::std::size_t;

/// This template exists only because C++ is stupid enough not to allow partial
/// function specialization. Long live boilerplate code!
template <typename, AllocMode>
class MemoryLifeManager;

template <typename T>
class MemoryLifeManager<T, AllocMode::SYNC> {
public:
    static inline cudaError_t
    malloc(device_ptr<T>* p, size_t nBytes) noexcept {
        return cudaMalloc(p, nBytes);
    }

    static inline cudaError_t
    free(device_ptr<T> p) noexcept {
        return cudaFree(p);
    }
};

template <typename T>
class MemoryLifeManager<T, AllocMode::ASYNC> {
public:
    static inline cudaError_t
    malloc(device_ptr<T>* p, size_t nBytes) noexcept {
        return cudaMallocAsync(p, nBytes);
    }

    static inline cudaError_t
    free(device_ptr<T> p) noexcept {
        return cudaFreeAsync(p);
    }
};

template <typename T>
class MemoryLifeManager<T, AllocMode::MANAGED> {
public:
    static inline cudaError_t
    malloc(device_ptr<T>* p, size_t nBytes) noexcept {
        return cudaMallocManaged(p, nBytes);
    }

    static inline cudaError_t
    free(device_ptr<T> p) noexcept {
        return cudaFree(p);
    }
};

template <typename T>
class MemoryLifeManager<T, AllocMode::HOST_PINNED> {
public:
    static inline cudaError_t
    malloc(device_ptr<T>* p, size_t nBytes) noexcept {
        return cudaMallocHost(p, nBytes);
    }

    static inline cudaError_t
    free(device_ptr<T> p) noexcept {
        return cudaFreeHost(p);
    }
};

template <typename T, AllocMode Mode>
class AllocatorBase {
protected:
    using MemMgr = MemoryLifeManager<T, Mode>;

public:
    using value_type      = T;
    using pointer         = device_ptr<T>;
    using const_pointer   = const_device_ptr<T>;
    using size_type       = cuda::std::size_t;
    using difference_type = cuda::std::ptrdiff_t;
    using is_always_equal = cuda::std::true_type; // stateless allocator

    [[nodiscard]] AllocatorBase() noexcept = default;

    template <typename U = T>
    [[nodiscard]] inline AllocatorBase(const AllocatorBase<U, Mode>&) noexcept {
    }

    [[nodiscard("MEMORY LEAK")]] pointer
    allocate(size_type n) const {
        pointer p;

        errors::check(MemMgr::malloc(&p, n * sizeof(T)));

        return p;
    }

    void
    deallocate(pointer p, size_type = 0) const {
        if (p) {
            errors::check(MemMgr::free(p));
        }
    }
};

} // namespace

template <typename T>
using DeviceAllocator = AllocatorBase<T, AllocMode::SYNC>;

template <typename T>
using DeviceAllocatorAsync = AllocatorBase<T, AllocMode::ASYNC>;

template <typename T>
using HostAllocatorManaged = AllocatorBase<T, AllocMode::MANAGED>;

template <typename T>
using HostAllocatorPinned = AllocatorBase<T, AllocMode::HOST_PINNED>;

using deallocationStatus_t = int;

namespace {

void
logDeallocationError(const std::exception& e) noexcept {
    std::cerr << "ERROR suppressed during device memory destructor call:\n"
              << e.what() << '\n';
}

/// unknown error type, exit
[[noreturn]] void
handleDeallocationError() noexcept {
    std::cerr << "UNKNOWN ERROR during device memory destruction - aborting";

    std::exit(EXIT_FAILURE);
}

template <typename E>
inline deallocationStatus_t
handleDeallocationError(const E&) noexcept;

template <>
inline deallocationStatus_t
handleDeallocationError<errors::DeviceOperationFailedError>(
    const errors::DeviceOperationFailedError& e) noexcept {

    logDeallocationError(e);
    return e.errCode;
}

template <>
inline deallocationStatus_t
handleDeallocationError<std::exception>(const std::exception& e) noexcept {

    logDeallocationError(e);
    return -1;
}

} // namespace

template <typename T, AllocMode Mode>
inline deallocationStatus_t
deallocateSafely(const AllocatorBase<T, Mode>& allocator, device_ptr<T> p,
                 cuda::std::size_t = 0) noexcept {

    try {
        allocator.deallocate(p);
        return cudaSuccess;
    } catch (const errors::DeviceOperationFailedError& e) {
        return handleDeallocationError(e);
    } catch (const std::exception& e) {
        return handleDeallocationError(e);
    } catch (...) {
        handleDeallocationError();
    }
}

namespace concepts {

template <typename T, typename Member>
concept HasMemberType = true;

template <typename A, typename T>
concept Allocator =
    requires(A a, cuda::std::size_t nBytes, device_ptr<T> p) {
        { a.allocate(nBytes) } -> std::same_as<device_ptr<T>>;
        { a.deallocate(p, nBytes) } -> std::same_as<void>;
    } and HasMemberType<A, typename A::value_type> and
    HasMemberType<A, typename A::pointer> and
    HasMemberType<A, typename A::size_type> and
    HasMemberType<A, typename A::difference_type> and
    HasMemberType<A, typename A::is_always_equal>;

template <typename A>
concept DeviceAllocator =
    SameAsAny<A, DeviceAllocator<typename A::value_type>,
              DeviceAllocatorAsync<typename A::value_type>>;

template <typename A>
concept HostAllocator =
    SameAsAny<A, HostAllocatorManaged<typename A::value_type>,
              HostAllocatorPinned<typename A::value_type>>;

} // namespace concepts

} // namespace allocator
} // namespace memory
} // namespace device
