#pragma once

#include "./concepts.hxx"
#include "./cuda_utils/exceptions.cuh"
#include "./launchers/functional.cuh"
#include "./launchers/launchers.cuh"
#include "./macros.hxx"
#include "./utils/text.hxx"
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
    DEFINE_CUDA_ERROR(DeviceBadAllocError, "Could not allocate memory on GPU")
    DEFINE_SIMPLE_ERROR(DeviceBadArrayNewLength, "Too big device array length")

    template <typename T>
    class DeviceAllocator {
    public:
        using value_type      = T;
        using pointer         = T*;
        using size_type       = std::size_t;
        using is_always_equal = std::true_type; // stateless allocator

        DeviceAllocator() = default;

        template <typename U>
        inline DeviceAllocator(const DeviceAllocator<U>&) noexcept {}

        [[nodiscard]] pointer allocate(size_type n);
        void                  deallocate(pointer p, size_type n);
        void                  deallocate(pointer p);
    };

    template <typename T>
    inline DeviceAllocator<T>::pointer DeviceAllocator<T>::allocate(size_type n) {
        const auto maxElemN = std::numeric_limits<size_type>::max() / sizeof(T);

        if (n > maxElemN)
            throw DeviceBadArrayNewLength();

        pointer    p;
        const auto status = cudaMalloc<T>(&p, n);

        DeviceBadAllocError::check(status);

        return p;
    }

    DEFINE_CUDA_ERROR(DeviceFreeError, "Could not free device memory")

    template <typename T>
    inline void DeviceAllocator<T>::deallocate(pointer p, size_type) {
        const auto status = cudaFree(p);
        DeviceFreeError::check(status);
    }

    template <typename T>
    inline void DeviceAllocator<T>::deallocate(pointer p) {
        deallocate(p, 0);
    }

    template <typename T>
    class DeviceMemoryLifetimeManager {
    public:
        using value_t = T;
        using pointer = T*;
        using size_t  = std::size_t;

        DeviceMemoryLifetimeManager(const DeviceMemoryLifetimeManager<T>&);
        DeviceMemoryLifetimeManager(size_t size);
        ~DeviceMemoryLifetimeManager();

        template <typename U = T>
        inline U* data() const noexcept;
        size_t    size() const noexcept;
        size_t    sizeBytes() const noexcept;

    private:
        DeviceMemoryLifetimeManager();
        DeviceMemoryLifetimeManager(size_t size, pointer&& pData);
        DeviceAllocator<value_t> mAllocator;

    protected:
        const size_t mSize  = 0;
        pointer      mpData = nullptr;
    };

    template <typename T>
    DeviceMemoryLifetimeManager<T>::DeviceMemoryLifetimeManager(const DeviceMemoryLifetimeManager<T>& other)
    : DeviceMemoryLifetimeManager{other.mSize} {}

    /// @brief Assumes that pData has already been allocated on a device
    /// @tparam T
    /// @param size
    /// @param pData
    template <typename T>
    inline DeviceMemoryLifetimeManager<T>::DeviceMemoryLifetimeManager(
        const size_t size, pointer&& pData)
    : mSize{size}, mpData{std::move(pData)} {}

    template <typename T>
    inline DeviceMemoryLifetimeManager<T>::DeviceMemoryLifetimeManager(const size_t size)
    : mSize{size}, mpData{mAllocator.allocate(sizeBytes())} {}

    template <typename T>
    inline DeviceMemoryLifetimeManager<T>::~DeviceMemoryLifetimeManager() {
        try {
            mAllocator.deallocate(mpData, sizeBytes());
            // no throwing inside the destructor
        } catch (const DeviceFreeError&) {
            std::cerr << "ERROR: could not deallocate device memory\n";
        } catch (...) {
            std::cerr << "UNKNOWN ERROR: could not deallocate device memory\n";
        }
    }

    template <typename T>
    template <typename U>
    inline U* DeviceMemoryLifetimeManager<T>::template data<U>() const noexcept {
        return mpData;
    }

    template <typename T>
    inline DeviceMemoryLifetimeManager<T>::size_t
    DeviceMemoryLifetimeManager<T>::size() const noexcept {
        return mSize;
    }

    template <typename T>
    inline DeviceMemoryLifetimeManager<T>::size_t
    DeviceMemoryLifetimeManager<T>::sizeBytes() const noexcept {
        return mSize * sizeof(T);
    }

    template <std::default_initializable T>
    class DeviceArr {
    public:
        DeviceArr(const DeviceArr<T>& other);
        DeviceArr(DeviceArr<T>&& other);
        DeviceArr(std::unique_ptr<DeviceMemoryLifetimeManager<T>>&& pMemMgr);
        /// @brief Does not initialize values in memory
        /// @param size
        DeviceArr(std::size_t size);
        DeviceArr(std::size_t size, T fillval);
        DeviceArr(std::size_t size, const T* fillval);
        DeviceArr(const T* arr, std::size_t size);
        DeviceArr(std::initializer_list<T> iniList);

        template <std::input_iterator InputIt>
        inline DeviceArr(InputIt begin, InputIt end);

        template <std::size_t N>
        inline DeviceArr(const std::array<T, N>& arr);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        inline DeviceArr(const std::vector<T, Alloc>& vec);

        // Getters
        std::size_t size() const noexcept;
        std::size_t sizeBytes() const noexcept;
        T*          data() const noexcept;

        template <std::default_initializable U = T>
        DeviceArr<U> copy() const;

        // Modifiers
        // cudaMemset sets bytes, so for sizeof(T) > 1 values representable
        // by >1 bytes cannot be set this way, hence need to implement fill
        void fill(T x);

        template <std::default_initializable U = T, concepts::MappingFn<T, U> F>
        DeviceArr<U> transform(F f = F());

        template <concepts::MappingFn<T> F>
        void transform_inplace(F f);

        // TODO reduce

        // Host <-> device
        std::string toString() const;
        void        print(const std::string& end = "\n", std::ostream& out = std::cout) const;

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        inline std::vector<T, Alloc> toHost() const;

        // Static methods
        template <std::size_t N>
        inline static DeviceArr fromArray(const std::array<T, N>& arr);

        static DeviceArr fromCArray(const T* arr, std::size_t size);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        inline static DeviceArr fromVector(const std::vector<T, Alloc>& vec);

        template <concepts::InputIter<T> InputIt>
        inline static DeviceArr fromIter(InputIt begin, InputIt end);

    private:
        DeviceArr();

    protected:
        // This is a unique_ptr to avoid double free when destructor is called
        // after std::move
        std::unique_ptr<DeviceMemoryLifetimeManager<T>> mpMemMgr;
    };

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(const DeviceArr<T>& other)
    : DeviceArr(other.size()) { std::cout << "copy array!\n"; }

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(DeviceArr<T>&& other) {
        mpMemMgr.swap(other.mpMemMgr);
    }

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(std::unique_ptr<DeviceMemoryLifetimeManager<T>>&& pMemMgr)
    : mpMemMgr{std::move(pMemMgr)} {}

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(const std::size_t size)
    : DeviceArr::DeviceArr(size, T{}) {}

    DEFINE_CUDA_ERROR(DeviceArrFillError, "Could not fill memory with initial value")

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(const std::size_t size, const T fillval)
    : DeviceArr{std::make_unique<DeviceMemoryLifetimeManager<T>>(size)} {
        fill(fillval);
    }

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(const std::size_t size, const T* const fillval)
    : DeviceArr{size, *fillval} {}

    DEFINE_CUDA_ERROR(DeviceArrToDeviceError, "Could not copy memory from host to device")

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(const T* const arr, const std::size_t size)
    : DeviceArr::DeviceArr(size) {
        const auto status = cudaMemcpy(
            mpMemMgr->data(), arr, sizeBytes(), cudaMemcpyHostToDevice);

        DeviceArrToDeviceError::check(status);
    }

    template <std::default_initializable T>
    inline DeviceArr<T>::DeviceArr(std::initializer_list<T> iniList)
    : DeviceArr{iniList.begin(), iniList.end()} {}

    template <std::default_initializable T>
    template <std::input_iterator InputIt>
    inline DeviceArr<T>::DeviceArr(InputIt begin, InputIt end)
    : DeviceArr{std::vector<T>(begin, end)} {}

    template <std::default_initializable T>
    template <typename Alloc>
    inline DeviceArr<T>::DeviceArr(const std::vector<T, Alloc>& vec)
    : DeviceArr::DeviceArr{vec.data(), vec.size()} {}

    template <std::default_initializable T>
    template <std::size_t N>
    inline DeviceArr<T>::DeviceArr(const std::array<T, N>& arr)
    : DeviceArr::DeviceArr{arr.data(), N} {}

    template <std::default_initializable T>
    inline std::size_t DeviceArr<T>::size() const noexcept {
        return mpMemMgr->size();
    }

    template <std::default_initializable T>
    inline std::size_t DeviceArr<T>::sizeBytes() const noexcept {
        return mpMemMgr->sizeBytes();
    }

    template <std::default_initializable T>
    inline T* DeviceArr<T>::data() const noexcept {
        return mpMemMgr->data();
    }

    DEFINE_CUDA_ERROR(DeviceArrCopyError, "Could not copy memory from device to device")

    template <std::default_initializable T>
    template <std::default_initializable U>
    inline DeviceArr<U> DeviceArr<T>::copy() const {
        DeviceArr<T> newArr(size());
        const auto   status = cudaMemcpy(newArr.data(), data(), sizeBytes(), cudaMemcpyDeviceToDevice);
        DeviceArrCopyError::check(status);

        return newArr;
    }

    DEFINE_CUDA_ERROR(DeviceArrToHostError, "Could not copy memory from device to host")

    template <std::default_initializable T>
    inline void DeviceArr<T>::fill(const T fillval) {
        try {
            launcher::fill(mpMemMgr->data(), fillval, size());
            cuda_utils::host::checkKernelLaunch("");
        } catch (const cuda_utils::host::CudaKernelLaunchError& e) {
            throw DeviceArrFillError(e.err);
        }
    }

    template <std::default_initializable T>
    template <std::default_initializable U, concepts::MappingFn<T, U> F>
    inline DeviceArr<U> DeviceArr<T>::transform(F f) {
        DeviceArr<U> newArr(size());
        launcher::map(newArr.data(), data(), size(), f);
        cuda_utils::host::checkKernelLaunch("map");

        return newArr;
    }

    template <std::default_initializable T>
    template <concepts::MappingFn<T> F>
    inline void DeviceArr<T>::transform_inplace(F f) {
        launcher::map(data(), data(), size(), f);
        cuda_utils::host::checkKernelLaunch("map");
    }

    template <std::default_initializable T>
    inline std::string DeviceArr<T>::toString() const {
        return utils::fmtVec(toHost());
    }

    template <std::default_initializable T>
    inline void DeviceArr<T>::print(const std::string& end, std::ostream& out) const {
        utils::printVec(toHost(), end, out);
    }

    template <std::default_initializable T>
    template <typename Alloc>
    inline std::vector<T, Alloc> DeviceArr<T>::toHost() const {
        // zeroed-out memory
        std::vector<T, Alloc> hostVec(size());

        const auto status = cudaMemcpy(
            hostVec.data(), mpMemMgr->data(), sizeBytes(), cudaMemcpyDeviceToHost);

        DeviceArrToHostError::check(status);

        return hostVec;
    }

    template <std::default_initializable T>
    template <std::size_t N>
    inline DeviceArr<T> DeviceArr<T>::fromArray(const std::array<T, N>& arr) {
        return {arr};
    }

    template <std::default_initializable T>
    inline DeviceArr<T> DeviceArr<T>::fromCArray(const T* const arr, std::size_t size) {
        return {arr, size};
    }

    template <std::default_initializable T>
    template <typename Alloc>
    inline DeviceArr<T> DeviceArr<T>::fromVector(const std::vector<T, Alloc>& vec) {
        return {vec};
    }

    template <std::default_initializable T>
    template <concepts::InputIter<T> InputIt>
    inline DeviceArr<T> DeviceArr<T>::fromIter(InputIt begin, InputIt end) {
        return {begin, end};
    }
} // namespace raii
