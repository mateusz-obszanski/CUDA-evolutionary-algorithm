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

namespace raii {
    template <typename T>
    class GpuArr {
        GpuArr();

    protected:
        std::size_t mLength = 0;
        T*          mVec    = nullptr;

    public:
        /// @brief Does not initialize values in memory
        /// @param length
        GpuArr(std::size_t length);
        GpuArr(std::size_t length, T fillval);
        GpuArr(const T arr[], std::size_t length);

        template <std::size_t N>
        GpuArr(const std::array<T, N>& arr);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        GpuArr(const std::vector<T, Alloc>& vec);

        ~GpuArr();

        size_t getLength() const noexcept;
        size_t getLengthBytes() const noexcept;

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        std::vector<T, Alloc> toHost() const;
    };

    DEFINE_SIMPLE_ERROR(GpuArrMallocError, "Could not allocate memory on GPU")

    template <typename T>
    GpuArr<T>::GpuArr(const std::size_t length) {
        this->length = length;

        const auto status = cudaMalloc<T>(&mVec, getLengthBytes());

        if (status != cudaSuccess)
            throw GpuArrMallocError();
    }

    DEFINE_SIMPLE_ERROR(GpuArrFillError, "Could not fill memory with initial value")

    template <typename T>
    GpuArr<T>::GpuArr(const std::size_t length, const T fillval) : GpuArr<T>::GpuArr{length} {
        const auto status = cudaMemset(mVec, fillval, getLengthBytes());

        if (status != cudaSuccess)
            throw GpuArrFillError();
    }

    DEFINE_SIMPLE_ERROR(GpuArrToGpuError, "Could not copy memory from host to device")

    template <typename T>
    GpuArr<T>::GpuArr(const T arr[], const std::size_t length) : GpuArr<T>::GpuArr{length} {
        const auto status = cudaMemcpy(mVec, arr, getLengthBytes(), cudaMemcpyHostToDevice);

        if (status != cudaSuccess)
            throw GpuArrToGpuError();
    }

    template <typename T>
    template <typename Alloc>
    GpuArr<T>::GpuArr(const std::vector<T, Alloc>& vec) : GpuArr<T>::GpuArr{&vec[0], vec.length()} {}

    template <typename T>
    template <std::size_t N>
    GpuArr<T>::GpuArr(const std::array<T, N>& arr) : GpuArr<T>::GpuArr{&arr[0], N} {}

    template <typename T>
    std::size_t GpuArr<T>::getLength() const noexcept {
        return mLength;
    }

    template <typename T>
    std::size_t GpuArr<T>::getLengthBytes() const noexcept {
        return mLength * sizeof(T);
    }

    DEFINE_SIMPLE_ERROR(GpuArrToHostError, "Could not copy memory from device to host")

    template <typename T>
    template <typename Alloc>
    std::vector<T, Alloc> GpuArr<T>::toHost() const {
        std::vector<T, Alloc> hostVec;
        hostVec.reserve(mLength);

        const auto status = cudaMemcpy(&hostVec[0], mVec, getLengthBytes());

        if (status != cudaSuccess)
            throw GpuArrToHostError();

        return hostVec;
    }

    template <typename T>
    GpuArr<T>::~GpuArr() {
        const auto status = cudaFree(mVec);

        if (status != cudaSuccess)
            std::cerr << "WARNING: could not free memory on device\n";
    }
} // namespace raii
