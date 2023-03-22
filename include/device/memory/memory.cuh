#pragma once

#include "../../errors.hxx"
#include "../../types/concepts.hxx"
#include "../iterator.cuh"
#include "../kernel_utils.cuh"
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

    device_ptr_out<T>   dst,
    device_ptr_in<T>    src,
    cuda::std::size_t   nElems,
    const cudaStream_t& stream = 0) {

    const auto status = cudaMemcpyAsync(
        dst, src, sizeof(T) * nElems, cudaMemcpyDeviceToDevice, stream);

    errors::check(status);
}

template <typename SrcT, ::types::concepts::ConstructibleButDifferentFrom<SrcT> DstT>
    requires ::types::concepts::Mutable<DstT>
inline void
copy(
    device_ptr_out<DstT> dst,
    device_ptr_in<SrcT>  src,
    cuda::std::size_t    nElems,
    const cudaStream_t&  stream = 0) {

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(nElems);

    kernel::copy<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT, 0, stream>>>(
        dst, src, nElems);

    errors::check();
}

namespace raii {

template <typename T>
concept DeviceStorable = std::copyable<T> and
                         std::default_initializable<T> and
                         ::types::concepts::Mutable<T>;

template <typename To, typename From>
concept DeviceCopyable = DeviceStorable<To> and
                         DeviceStorable<From> and
                         std::constructible_from<From, To>;

template <
    DeviceStorable T,

    template <typename TT>
    typename Allocator = allocator::DeviceAllocator>

    requires allocator::concepts::Allocator<Allocator<T>, T>
class Memory {
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
    Memory() = delete;

    [[nodiscard]] Memory(size_type size, Allocator<T> allocator = Allocator<T>())
    : mSize{size}, mAllocator(allocator), mpData{mAllocator.allocate(size)} {}

    template <DeviceCopyable<T> U>
    [[nodiscard]] Memory(const Memory<U, Allocator>& other)
    : Memory(other.mSize, other.mAllocator) {
        other.template copy<T>(mpData);
    }

    [[nodiscard]] Memory(Memory<T, Allocator>&& other)
    : mAllocator{other.mAllocator}, mSize{other.mSize}, mpData{other.mpData} {
        other.mpData = nullptr;
    }

    template <DeviceCopyable<T> U>
    [[nodiscard]] Memory(Memory<U, Allocator>&& other)
    : Memory(other) {
        other.mpData = nullptr;
    }

    ~Memory() noexcept {
        allocator::deallocateSafely(mAllocator, mpData);
    }

    [[nodiscard]] inline const_pointer
    data() const noexcept { return mpData; }

    [[nodiscard]] inline pointer
    data() noexcept { return mpData; }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return mSize; }

    [[nodiscard]] constexpr size_type
    sizeBytes() const noexcept { return mSize * sizeof(T); }

    template <DeviceCopyable<T> U = T>
    void
    copy(device_ptr<U> pTo, const cudaStream_t& stream = 0) const {
        memory::copy(pTo, mpData, mSize, stream);
    }

    template <DeviceCopyable<T> U = T>
    void
    copy(Memory<U, Allocator>& dstOther, const cudaStream_t& stream = 0) const {
        copy(dstOther.data(), stream);
    }

    template <
        DeviceCopyable<T> U = T,

        template <typename TT>
        typename NewAllocator = Allocator>

        requires allocator::concepts::Allocator<NewAllocator<U>, U> and
                 ::types::concepts::DifferentFrom<NewAllocator<U>, allocator::HostAllocatorPinned<U>>
    [[nodiscard]] Memory<U, NewAllocator>
    copy(const cudaStream_t& stream = 0) const {
        Memory<U, NewAllocator> newMem(mSize);

        copy(newMem, stream);

        return newMem;
    }

    template <
        DeviceCopyable<T> U = T,

        template <typename TT>
        typename NewAllocator = Allocator>

        requires std::same_as<NewAllocator<U>, allocator::HostAllocatorPinned<U>>
    [[nodiscard]] Memory<U, allocator::HostAllocatorPinned>
    copy(const cudaStream_t& stream = 0) const {
        // TODO copy from device to host pinned memory instead of normal device-device copy
        ::errors::throwNotImplemented();
    }

    [[nodiscard]] pointer
    begin_ptr() noexcept {
        return mpData;
    }

    [[nodiscard]] const_pointer
    begin_ptr() const noexcept {
        return mpData;
    }

    [[nodiscard]] pointer
    end_ptr() noexcept {
        return mpData + mSize;
    }

    [[nodiscard]] const_pointer
    end_ptr() const noexcept {
        return mpData + mSize;
    }

    [[nodiscard]] const_pointer
    cbegin_ptr() const noexcept {
        return begin_ptr();
    }

    [[nodiscard]] const_pointer
    cend_ptr() const noexcept {
        return end_ptr();
    }

    [[nodiscard]] pointer
    rbegin_ptr() noexcept {
        return end_ptr() - 1;
    }

    [[nodiscard]] const_pointer
    rbegin_ptr() const noexcept {
        return end_ptr() - 1;
    }

    [[nodiscard]] pointer
    rend_ptr() noexcept {
        return begin_ptr() - 1;
    }

    [[nodiscard]] const_pointer
    rend_ptr() const noexcept {
        return begin_ptr() - 1;
    }

    [[nodiscard]] const_pointer
    crbegin_ptr() const noexcept {
        return rbegin_ptr();
    }

    [[nodiscard]] const_pointer
    crend_ptr() const noexcept {
        return rend_ptr();
    }

    [[nodiscard]] iterator
    begin() noexcept { return {begin_ptr()}; }

    [[nodiscard]] const_iterator
    begin() const noexcept { return {begin_ptr()}; }

    [[nodiscard]] iterator
    end() noexcept { return {end_ptr()}; }

    [[nodiscard]] const_iterator
    end() const noexcept { return {end_ptr()}; }

    [[nodiscard]] const_iterator
    cbegin() const noexcept { return {cbegin_ptr()}; }

    [[nodiscard]] const_iterator
    cend() const noexcept { return {cend_ptr()}; }

    [[nodiscard]] reverse_iterator
    rbegin() noexcept { return {rbegin_ptr()}; }

    [[nodiscard]] const_reverse_iterator
    rbegin() const noexcept { return {rbegin_ptr()}; }

    [[nodiscard]] reverse_iterator
    rend() noexcept { return {rend_ptr()}; }

    [[nodiscard]] const_reverse_iterator
    rend() const noexcept { return {rend_ptr()}; }

    [[nodiscard]] const_reverse_iterator
    crbegin() const noexcept { return {crbegin_ptr()}; }

    [[nodiscard]] const_reverse_iterator
    crend() const noexcept { return {crend_ptr()}; }

    [[nodiscard]] thrust_iterator
    begin_thrust() noexcept { return begin().baseThrust(); }

    [[nodiscard]] thrust_const_iterator
    begin_thrust() const noexcept { return begin().baseThrust(); }

    [[nodiscard]] thrust_iterator
    end_thrust() noexcept { return end().baseThrust(); }

    [[nodiscard]] thrust_const_iterator
    end_thrust() const noexcept { return end().baseThrust(); }

    [[nodiscard]] thrust_const_iterator
    cbegin_thrust() const noexcept { return cbegin().baseThrust(); }

    [[nodiscard]] thrust_const_iterator
    cend_thrust() const noexcept { return cend().baseThrust(); }

    [[nodiscard]] thrust_reverse_iterator
    rbegin_thrust() noexcept { return rbegin().baseThrust(); }

    [[nodiscard]] thrust_const_reverse_iterator
    rbegin_thrust() const noexcept { return rbegin().baseThrust(); }

    [[nodiscard]] thrust_reverse_iterator
    rend_thrust() noexcept { return rend().baseThrust(); }

    [[nodiscard]] thrust_const_reverse_iterator
    rend_thrust() const noexcept { return rend().baseThrust(); }

    [[nodiscard]] thrust_const_reverse_iterator
    crbegin_thrust() const noexcept { return crbegin().toThrust(); }

    [[nodiscard]] thrust_const_reverse_iterator
    crend_thrust() const noexcept { return crend().toThrust(); }

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

template <DeviceStorable T>
using DeviceMemory = Memory<T, allocator::DeviceAllocator>;

template <DeviceStorable T>
using DeviceMemoryAsync = Memory<T, allocator::DeviceAllocatorAsync>;

template <DeviceStorable T>
using HostMemoryManaged = Memory<T, allocator::HostAllocatorManaged>;

} // namespace raii
} // namespace memory
} // namespace device
