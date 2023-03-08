#pragma once

#include "./cuda_utils/exceptions.cuh"
#include "./macros.hxx"
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <vector>

// TODO exception hierarchy
// TODO parent CUDA operation error with last error message

namespace raii {
    DEFINE_CUDA_ERROR(DeviceAllocError, "Could not allocate memory on GPU")

    template <typename T>
    class DeviceMemoryLifeManager {
        DeviceMemoryLifeManager();
        DeviceMemoryLifeManager(std::size_t size, T*&& pData);

    protected:
        const std::size_t mSize  = 0;
        T*                mpData = nullptr;

    public:
        DeviceMemoryLifeManager(std::size_t size);
        ~DeviceMemoryLifeManager();

        T*     data() const noexcept;
        size_t size() const noexcept;
        size_t sizeBytes() const noexcept;
    };

    /// @brief Assumes that pData has already been allocated on a device
    /// @tparam T
    /// @param size
    /// @param pData
    template <typename T>
    DeviceMemoryLifeManager<T>::DeviceMemoryLifeManager(const std::size_t size, T*&& pData)
    : mSize{size}, mpData{std::move(pData)} {}

    template <typename T>
    DeviceMemoryLifeManager<T>::DeviceMemoryLifeManager(const std::size_t size)
    : mSize{size} {
        const auto status = cudaMalloc(
            reinterpret_cast<void**>(&mpData), sizeBytes());

        DeviceAllocError::check(status);
    }

    template <typename T>
    DeviceMemoryLifeManager<T>::~DeviceMemoryLifeManager() {
        const auto status = cudaFree(mpData);

        if (status != cudaSuccess)
            std::cerr << "ERROR: could not free memory on device\n";
    }

    template <typename T>
    T* DeviceMemoryLifeManager<T>::data() const noexcept {
        return mpData;
    }

    template <typename T>
    std::size_t DeviceMemoryLifeManager<T>::size() const noexcept {
        return mSize;
    }

    template <typename T>
    std::size_t DeviceMemoryLifeManager<T>::sizeBytes() const noexcept {
        return mSize * sizeof(T);
    }

    template <std::default_initializable T>
    class DeviceArr {
        DeviceArr();

    protected:
        const std::unique_ptr<DeviceMemoryLifeManager<T>> mpMemMgr;

    public:
        DeviceArr(std::unique_ptr<DeviceMemoryLifeManager<T>>&& pMemMgr);
        /// @brief Does not initialize values in memory
        /// @param size
        DeviceArr(std::size_t size);
        DeviceArr(std::size_t size, T fillval);
        DeviceArr(std::size_t size, T* fillval);
        DeviceArr(const T* arr, std::size_t size);
        DeviceArr(std::initializer_list<T> iniList);

        template <typename InputIt>
        DeviceArr(InputIt begin, InputIt end);

        template <std::size_t N>
        DeviceArr(const std::array<T, N>& arr);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        DeviceArr(const std::vector<T, Alloc>& vec);

        size_t size() const noexcept;
        size_t sizeBytes() const noexcept;

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        std::vector<T, Alloc> toHost() const;

        template <std::size_t N>
        static DeviceArr fromArray(const std::array<T, N>& arr);

        static DeviceArr fromCArray(const T* arr, std::size_t size);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        static DeviceArr fromVector(const std::vector<T, Alloc>& vec);

        template <typename InputIt>
        static DeviceArr fromIter(InputIt begin, InputIt end);
    };

    template <std::default_initializable T>
    DeviceArr<T>::DeviceArr(std::unique_ptr<DeviceMemoryLifeManager<T>>&& pMemMgr)
    : mpMemMgr{std::move(pMemMgr)} {}

    template <std::default_initializable T>
    DeviceArr<T>::DeviceArr(const std::size_t size)
    : DeviceArr{std::make_unique<DeviceMemoryLifeManager<T>>(size)} {}

    DEFINE_CUDA_ERROR(DeviceArrFillError, "Could not fill memory with initial value")

    template <std::default_initializable T>
    DeviceArr<T>::DeviceArr(const std::size_t size, const T fillval)
    : DeviceArr::DeviceArr{size} {
        const auto status = cudaMemset(mpMemMgr->data(), fillval, sizeBytes());

        DeviceArrFillError::check(status);
    }

    template <std::default_initializable T>
    DeviceArr<T>::DeviceArr(const std::size_t size, T* const fillval)
    : DeviceArr{size, *fillval} {}

    DEFINE_CUDA_ERROR(DeviceArrToDeviceError, "Could not copy memory from host to device")

    template <std::default_initializable T>
    DeviceArr<T>::DeviceArr(const T* const arr, const std::size_t size)
    : DeviceArr::DeviceArr(size) {
        const auto status = cudaMemcpy(
            mpMemMgr->data(), arr, sizeBytes(), cudaMemcpyHostToDevice);

        DeviceArrToDeviceError::check(status);
    }

    template <std::default_initializable T>
    DeviceArr<T>::DeviceArr(std::initializer_list<T> iniList)
    : DeviceArr{iniList.begin(), iniList.end()} {}

    template <std::default_initializable T>
    template <typename InputIt>
    DeviceArr<T>::DeviceArr(InputIt begin, InputIt end)
    : DeviceArr{std::vector<T>(begin, end)} {}

    template <std::default_initializable T>
    template <typename Alloc>
    DeviceArr<T>::DeviceArr(const std::vector<T, Alloc>& vec)
    : DeviceArr::DeviceArr{vec.data(), vec.size()} {}

    template <std::default_initializable T>
    template <std::size_t N>
    DeviceArr<T>::DeviceArr(const std::array<T, N>& arr)
    : DeviceArr::DeviceArr{arr.data(), N} {}

    template <std::default_initializable T>
    std::size_t DeviceArr<T>::size() const noexcept {
        return mpMemMgr->size();
    }

    template <std::default_initializable T>
    std::size_t DeviceArr<T>::sizeBytes() const noexcept {
        return mpMemMgr->sizeBytes();
    }

    DEFINE_CUDA_ERROR(DeviceArrToHostError, "Could not copy memory from device to host")

    template <std::default_initializable T>
    template <typename Alloc>
    std::vector<T, Alloc> DeviceArr<T>::toHost() const {
        // zeroed-out memory
        std::vector<T, Alloc> hostVec(size());

        const auto status = cudaMemcpy(
            hostVec.data(), mpMemMgr->data(), sizeBytes(), cudaMemcpyDeviceToHost);

        DeviceArrToHostError::check(status);

        return hostVec;
    }

    template <std::default_initializable T>
    template <std::size_t N>
    DeviceArr<T> DeviceArr<T>::fromArray(const std::array<T, N>& arr) {
        return {arr};
    }

    template <std::default_initializable T>
    DeviceArr<T> DeviceArr<T>::fromCArray(const T* arr, std::size_t size) {
        return {arr, size};
    }

    template <std::default_initializable T>
    template <typename Alloc>
    DeviceArr<T> DeviceArr<T>::fromVector(const std::vector<T, Alloc>& vec) {
        return {vec};
    }

    template <std::default_initializable T>
    template <typename InputIt>
    DeviceArr<T> DeviceArr<T>::fromIter(InputIt begin, InputIt end) {
        return {begin, end};
    }
} // namespace raii
