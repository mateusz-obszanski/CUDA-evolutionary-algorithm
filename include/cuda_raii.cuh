#pragma once

#include "./macros.hxx"
#include <array>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <vector>

// TODO exception hierarchy
// TODO parent CUDA operation error with last error message

namespace raii {
    DEFINE_SIMPLE_ERROR(DeviceAllocError, "Could not allocate memory on GPU")

    template <typename T>
    class DeviceMemoryLifeManager {
        DeviceMemoryLifeManager();

    protected:
        const std::size_t mSize  = 0;
        T*                mpData = nullptr;

    public:
        DeviceMemoryLifeManager(std::size_t size);
        DeviceMemoryLifeManager(T*&& pData, std::size_t size);
        ~DeviceMemoryLifeManager();

        T*     getPtr() const noexcept;
        size_t getSize() const noexcept;
        size_t getSizeBytes() const noexcept;
    };

    template <typename T>
    DeviceMemoryLifeManager<T>::DeviceMemoryLifeManager(const std::size_t size)
    : mSize{size} {
        const auto status = cudaMalloc(
            reinterpret_cast<void**>(&mpData), getSizeBytes());

        if (status != cudaSuccess)
            throw DeviceAllocError();
    }

    template <typename T>
    DeviceMemoryLifeManager<T>::DeviceMemoryLifeManager(T*&& pData, const std::size_t size)
    : mSize{size}, mpData{std::move(pData)} {}

    template <typename T>
    DeviceMemoryLifeManager<T>::~DeviceMemoryLifeManager() {
        const auto status = cudaFree(mpData);

        if (status != cudaSuccess)
            std::cerr << "WARNING: could not free memory on device\n";
    }

    template <typename T>
    T* DeviceMemoryLifeManager<T>::getPtr() const noexcept {
        return mpData;
    }

    template <typename T>
    std::size_t DeviceMemoryLifeManager<T>::getSize() const noexcept {
        return mSize;
    }

    template <typename T>
    std::size_t DeviceMemoryLifeManager<T>::getSizeBytes() const noexcept {
        return mSize * sizeof(T);
    }

    template <typename T>
    class DeviceArr {
        DeviceArr();

    protected:
        const DeviceMemoryLifeManager<T> mMemMgr;

    public:
        DeviceArr(DeviceMemoryLifeManager<T>& memMgr);
        DeviceArr(DeviceMemoryLifeManager<T>&& memMgr);
        /// @brief Does not initialize values in memory
        /// @param length
        DeviceArr(std::size_t length);
        DeviceArr(std::size_t length, T fillval);
        DeviceArr(std::size_t length, T* fillval);
        DeviceArr(const T* arr, std::size_t length);

        template <std::size_t N>
        DeviceArr(const std::array<T, N>& arr);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        DeviceArr(const std::vector<T, Alloc>& vec);

        size_t getLength() const noexcept;
        size_t getLengthBytes() const noexcept;

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        std::vector<T, Alloc> toHost() const;
    };

    template <typename T>
    DeviceArr<T>::DeviceArr(DeviceMemoryLifeManager<T>& memMgr) : mMemMgr{memMgr} {}

    template <typename T>
    DeviceArr<T>::DeviceArr(DeviceMemoryLifeManager<T>&& memMgr)
    : mMemMgr{std::move(memMgr)} {}

    template <typename T>
    DeviceArr<T>::DeviceArr(const std::size_t length)
    : DeviceArr{DeviceMemoryLifeManager<T>{length}} {}

    DEFINE_SIMPLE_ERROR(GpuArrFillError, "Could not fill memory with initial value")

    template <typename T>
    DeviceArr<T>::DeviceArr(const std::size_t length, const T fillval)
    : DeviceArr::DeviceArr{length} {
        const auto status = cudaMemset(mMemMgr.getPtr(), fillval, getLengthBytes());

        if (status != cudaSuccess)
            throw GpuArrFillError();
    }

    template <typename T>
    DeviceArr<T>::DeviceArr(const std::size_t length, T* const fillval)
    : DeviceArr{length, *fillval} {}

    DEFINE_SIMPLE_ERROR(GpuArrToGpuError, "Could not copy memory from host to device")

    template <typename T>
    DeviceArr<T>::DeviceArr(const T* const arr, const std::size_t length)
    : DeviceArr::DeviceArr{length} {
        const auto status = cudaMemcpy(
            mMemMgr.getPtr(), arr, getLengthBytes(), cudaMemcpyHostToDevice);

        if (status != cudaSuccess)
            throw GpuArrToGpuError();
    }

    template <typename T>
    template <typename Alloc>
    DeviceArr<T>::DeviceArr(const std::vector<T, Alloc>& vec)
    : DeviceArr::DeviceArr{&vec[0], vec.length()} {}

    template <typename T>
    template <std::size_t N>
    DeviceArr<T>::DeviceArr(const std::array<T, N>& arr)
    : DeviceArr::DeviceArr{&arr[0], N} {}

    template <typename T>
    std::size_t DeviceArr<T>::getLength() const noexcept {
        return mMemMgr.getSize();
    }

    template <typename T>
    std::size_t DeviceArr<T>::getLengthBytes() const noexcept {
        return mMemMgr.getSizeBytes();
    }

    DEFINE_SIMPLE_ERROR(GpuArrToHostError, "Could not copy memory from device to host")

    template <typename T>
    template <typename Alloc>
    std::vector<T, Alloc> DeviceArr<T>::toHost() const {
        std::vector<T, Alloc> hostVec;
        hostVec.reserve(getLength());

        const auto status = cudaMemcpy(
            &hostVec[0], mMemMgr.getPtr(), getLengthBytes(), cudaMemcpyDeviceToHost);

        if (status != cudaSuccess)
            throw GpuArrToHostError();

        return hostVec;
    }
} // namespace raii
