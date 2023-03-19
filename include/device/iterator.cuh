#pragma once

#include "./types/types.cuh"
#include <cuda/std/iterator>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace device {
namespace iterator {

template <typename T>
class DeviceBaseIterator {
public:
    using iterator_category = cuda::std::bidirectional_iterator_tag;
    using difference_type   = cuda::std::ptrdiff_t;
    using value_type        = T;
    using pointer           = device_ptr<T>;
    using reference         = T&;

    [[nodiscard]] __host__ __device__
    DeviceBaseIterator(pointer p) : m_ptr{p} {}

    [[nodiscard]] __host__ __device__ pointer
    base() const noexcept {
        return m_ptr;
    }

    [[nodiscard]] __host__ __device__
    operator pointer() const noexcept {
        return m_ptr;
    }

    [[nodiscard]] __host__ __device__ reference
    operator*() const { return *m_ptr; }

    [[nodiscard]] __host__ __device__ pointer
    operator->() { return m_ptr; }

    friend __host__ __device__ bool
    operator<=>(
        DeviceBaseIterator const& lhs, DeviceBaseIterator const& rhs) = default;

protected:
    pointer m_ptr;
};

template <typename T>
class DeviceIterator : public DeviceBaseIterator<T> {
public:
    // ++this
    __host__ __device__ DeviceIterator&
    operator++() {
        ++(this->m_ptr);
        return *this;
    }

    // this++
    __host__ __device__ DeviceIterator
    operator++(int) {
        DeviceIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    // --this
    __host__ __device__ DeviceIterator&
    operator--() {
        --(this->m_ptr);
        return *this;
    }

    // this--
    __host__ __device__ DeviceIterator
    operator--(int) {
        DeviceIterator tmp = *this;
        --(*this);
        return tmp;
    }

    [[nodiscard]] __host__ __device__ DeviceIterator
    operator+(int x) const {
        return {this->m_ptr + x};
    }

    [[nodiscard]] __host__ __device__ DeviceIterator
    operator-(int x) const {
        return {this->m_ptr - x};
    }

    [[nodiscard]] __host__ __device__
        thrust::device_ptr<T>
        baseThrust() const noexcept {

        return thrust::device_pointer_cast(this->m_ptr);
    }
};

template <typename T>
using DeviceConstIterator = DeviceIterator<const T>;

template <typename T>
class DeviceReverseIterator : public DeviceBaseIterator<T> {
public:
    // ++this
    __host__ __device__ DeviceReverseIterator&
    operator++() {
        --(this->m_ptr);
        return *this;
    }

    // this++
    __host__ __device__ DeviceReverseIterator
    operator++(int) {
        DeviceReverseIterator tmp = *this;
        --(*this);
        return tmp;
    }

    // --this
    __host__ __device__ DeviceReverseIterator&
    operator--() {
        ++(this->m_ptr);
        return *this;
    }

    // this--
    __host__ __device__ DeviceReverseIterator
    operator--(int) {
        DeviceReverseIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    [[nodiscard]] __host__ __device__ DeviceReverseIterator
    operator+(int x) const {
        return {this->m_ptr - x};
    }

    [[nodiscard]] __host__ __device__ DeviceReverseIterator
    operator-(int x) const {
        return {this->m_ptr + x};
    }

    [[nodiscard]] __host__ __device__
        thrust::reverse_iterator<thrust::device_ptr<T>>
        baseThrust() const {

        return thrust::make_reverse_iterator(thrust::device_pointer_cast(this->m_ptr));
    }
};

template <typename T>
using DeviceConstReverseIterator = DeviceReverseIterator<const T>;

} // namespace iterator
} // namespace device
