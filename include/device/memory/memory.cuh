#pragma once

#include "../../errors.hxx"
#include "../../types/concepts.hxx"
#include "../iterator.cuh"
#include "../kernel_utils.cuh"
#include "../types/concept.cuh"
#include "./allocator.cuh"
#include "./kernels.cuh"
#include <concepts>
#include <cuda/std/cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector>

namespace device {
namespace memory {

template <typename T, std::same_as<T> U = T>
inline void
copy(

    device_ptr_out<T> dst,
    device_ptr_in<T>  src,
    cuda::std::size_t nElems) {

    const auto status = cudaMemcpy(
        dst, src, sizeof(T) * nElems, cudaMemcpyDeviceToDevice);

    errors::check(status);
}

template <typename SrcT, ::types::concepts::ConstructibleButDifferent<SrcT> DstT>
inline void
copy(
    device_ptr_out<DstT> dst,
    device_ptr_in<SrcT>  src,
    cuda::std::size_t    nElems) {

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(nElems);

    kernel::copy<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT>>>(
        dst, src, nElems);

    errors::check();
}

namespace raii {

template <
    typename T,

    template <typename TT>
    typename Allocator = allocator::DeviceAllocator>

    requires types::concepts::DeviceAllocator<Allocator<T>, T>
class DeviceMemory {
public:
    using allocator_type                = Allocator<T>;
    using value_type                    = allocator_type::value_type;
    using pointer                       = allocator_type::pointer;
    using const_pointer                 = allocator_type::const_pointer;
    using iterator                      = device::iterator::DeviceIterator<T>;
    using const_iterator                = device::iterator::DeviceConstIterator<T>;
    using reverse_iterator              = device::iterator::DeviceReverseIterator<T>;
    using const_reverse_iterator        = device::iterator::DeviceConstReverseIterator<T>;
    using thrust_pointer                = thrust::device_ptr<T>;
    using const_thrust_pointer          = thrust::device_ptr<const T>;
    using thrust_iterator               = thrust_pointer;
    using thrust_const_iterator         = const_thrust_pointer;
    using thrust_reverse_iterator       = thrust::reverse_iterator<thrust_pointer>;
    using thrust_const_reverse_iterator = thrust::reverse_iterator<const_thrust_pointer>;
    using size_type                     = allocator_type::size_type;

protected:
    [[no_unique_address]] allocator_type mAllocator;
    const size_type                      mSize  = 0;
    pointer                              mpData = nullptr;

public:
    DeviceMemory() = delete;

    [[nodiscard]] DeviceMemory(size_type size, Allocator<T> allocator = Allocator<T>())
    : mSize{size}, mAllocator(allocator), mpData{mAllocator.allocate(size)} {}

    template <std::constructible_from<T> U>
    [[nodiscard]] DeviceMemory(const DeviceMemory<U, Allocator>& other)
    : DeviceMemory(other.mSize, other.mAllocator) {
        other.template copy<T>(mpData);
    }

    ~DeviceMemory() noexcept {
        allocator::deallocateSafely(mAllocator, mpData);
    }

    [[nodiscard]] inline device_ptr<T>
    data() const noexcept { return mpData; }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return mSize; }

    [[nodiscard]] constexpr size_type
    sizeBytes() const noexcept { return mSize * sizeof(T); }

    template <std::constructible_from<T> U = T>
    void
    copy(device_ptr<U> pTo) const {
        memory::copy(pTo, mpData, mSize);
    }

    template <std::constructible_from<T> U = T>
    void
    copy(DeviceMemory<U, Allocator>& dstOther) const {
        copy(dstOther.data());
    }

    template <
        std::constructible_from<T> U = T,

        template <typename TT>
        typename NewAllocator = Allocator>

        requires(
            types::concepts::DeviceAllocator<NewAllocator<U>, U> and
            not std::same_as<NewAllocator<U>, allocator::DeviceAllocatorHostPinned<U>>)
    [[nodiscard]] DeviceMemory<U, NewAllocator> copy() const {
        DeviceMemory<U, NewAllocator> newMem(mSize);

        copy(newMem);

        return newMem;
    }

    template <
        std::constructible_from<T> U = T,

        template <typename TT>
        typename NewAllocator = Allocator>

        requires std::same_as<NewAllocator<U>, allocator::DeviceAllocatorHostPinned<U>>
    [[nodiscard]] DeviceMemory<U, allocator::DeviceAllocatorHostPinned>
    copy() const {
        // TODO copy from device to host pinned memory instead of normal device-device copy
        ::errors::throwNotImplemented();
    }

    [[nodiscard]] iterator
    begin() noexcept { return {mpData}; }

    [[nodiscard]] iterator
    end() noexcept { return {mpData + mSize}; }

    [[nodiscard]] const_iterator
    cbegin() const noexcept { return begin(); }

    [[nodiscard]] const_iterator
    cend() const noexcept { return end(); }

    [[nodiscard]] reverse_iterator
    rbegin() noexcept { return {end() - 1}; }

    [[nodiscard]] reverse_iterator
    rend() noexcept { return {begin() - 1}; }

    [[nodiscard]] const_reverse_iterator
    crbegin() const noexcept { return rbegin(); }

    [[nodiscard]] const_reverse_iterator
    crend() const noexcept { return rend(); }

    [[nodiscard]] thrust_iterator
    begin_thrust() noexcept { return begin().baseThrust(); }

    [[nodiscard]] thrust_iterator
    end_thrust() noexcept { return end().baseThrust(); }

    [[nodiscard]] thrust_const_iterator
    cbegin_thrust() noexcept { return cbegin().baseThrust(); }

    [[nodiscard]] thrust_const_iterator
    cend_thrust() noexcept { return cend().baseThrust(); }

    [[nodiscard]] thrust_reverse_iterator
    rbegin_thrust() noexcept { return rbegin().baseThrust(); }

    [[nodiscard]] thrust_reverse_iterator
    rend_thrust() noexcept { return rend().baseThrust(); }

    [[nodiscard]] thrust_const_reverse_iterator
    crbegin_thrust() noexcept { return crbegin().toThrust(); }

    [[nodiscard]] thrust_const_reverse_iterator
    crend_thrust() noexcept { return crend().toThrust(); }

    template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
    [[nodiscard]] inline std::vector<T, HostAlloc>
    toHost() const {
        std::vector<T, HostAlloc> hostVec(size());

        const auto status = cudaMemcpy(
            hostVec.data(), mpData, sizeBytes(), cudaMemcpyDeviceToHost);

        errors::check(status);

        return hostVec;
    }

    template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
    [[nodiscard]] inline std::string
    toString() const {
        return text::fmtVec(toHost<HostAlloc>());
    }

    template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
    inline void
    print() const {
        std::cout << toString<HostAlloc>() << '\n';
    }
};

} // namespace raii
} // namespace memory
} // namespace device
