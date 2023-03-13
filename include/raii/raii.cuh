#pragma once

#include "../concepts.hxx"
#include "../cuda_ops.cuh"
#include "../cuda_utils/exceptions.cuh"
#include "../exceptions.hxx"
#include "../launchers/functional.cuh"
#include "../launchers/launchers.cuh"
#include "../macros.hxx"
#include "../utils/text.hxx"
#include "./life.cuh"
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
    // helper concept
    // concepts::Addable/Subtractable and std::convertible_to<std::size_t>
    // for fancy counter counstructors
    template <typename T>
    concept DeviceArrValue =
        std::default_initializable<T> and
        concepts::Addable<T> and
        concepts::Subtractable<T> and
        std::convertible_to<T, std::size_t> and
        concepts::WeaklyComparable<T> and
        launcher::CountingAble<T>;

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate = allocator::DeviceAllocatorD1,

        typename... AllocatorParams>
    class DeviceArr {
    private:
        [[nodiscard]] DeviceArr();

    protected:
        using allocator_type = AllocatorTemplate<T, AllocatorParams...>;
        using mem_life_t     = life::DeviceMemoryLife<T, allocator_type>;

        // This is a unique_ptr to avoid double free when destructor is called
        // after std::move
        std::unique_ptr<mem_life_t> mpMemLife;

    public:
        using cls        = DeviceArr<T, AllocatorTemplate, AllocatorParams...>;
        using size_type  = mem_life_t::size_type;
        using value_type = T;

        [[nodiscard]] DeviceArr(const cls& other);
        [[nodiscard]] DeviceArr(cls&& other);
        [[nodiscard]] DeviceArr(std::unique_ptr<mem_life_t>&& pMemLife);

        /// @brief Does not initialize values in memory
        /// @param size
        [[nodiscard]] DeviceArr(size_type size);
        [[nodiscard]] DeviceArr(size_type size, T fillval);
        [[nodiscard]] DeviceArr(size_type size, const T* fillval);
        [[nodiscard]] DeviceArr(const host_ptr<T> arr, size_type size);
        [[nodiscard]] DeviceArr(std::initializer_list<T> iniList);

        template <std::input_iterator InputIt>
        [[nodiscard]] inline DeviceArr(InputIt begin, InputIt end);

        template <std::size_t N>
        [[nodiscard]] inline DeviceArr(const std::array<T, N>& arr);

        template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
        [[nodiscard]] inline DeviceArr(const std::vector<T, HostAlloc>& vec);

        // Getters
        [[nodiscard]] size_type
        size() const noexcept;

        [[nodiscard]] size_type
        sizeBytes() const noexcept;

        [[nodiscard]] device_ptr<T>
        data() const noexcept;

        template <DeviceArrValue U = T>
        [[nodiscard]] DeviceArr<U, AllocatorTemplate, AllocatorParams...>
        copy() const;

        // Modifiers
        // cudaMemset sets bytes, so for sizeof(T) > 1 values representable
        // by >1 bytes cannot be set this way, hence need to implement fill
        void
        fill(T x);

        template <DeviceArrValue U = T, concepts::MappingFn<T, U> F>
        [[nodiscard]] DeviceArr<U, AllocatorTemplate, AllocatorParams...>
        transform(F f = F()) const;

        template <concepts::MappingFn<T> F>
        void
        transform_inplace(F f);

        template <
            launcher::Reductible       Acc = T,
            concepts::Reductor<T, Acc> F   = cuda_ops::Add2<Acc, T>>
        [[nodiscard]] inline Acc
        reduce(F f = cuda_ops::Add2<Acc, T>()) const;

        // Host <-> device
        [[nodiscard]] std::string
        toString() const;

        void
        print(
            const std::string& end = "\n",
            std::ostream&      out = std::cout) const;

        template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
        [[nodiscard]] inline std::vector<T, HostAlloc>
        toHost() const;

        // Static methods
        [[nodiscard]] inline static DeviceArr
        fromDevicePtr(device_ptr<T>&&, size_type);

        template <std::size_t N>
        [[nodiscard]] inline static DeviceArr
        fromArray(const std::array<T, N>& arr);

        [[nodiscard]] static DeviceArr
        fromRawArray(const T* arr, std::size_t size);

        template <typename HostAlloc = std::pmr::polymorphic_allocator<T>>
        [[nodiscard]] inline static DeviceArr
        fromVector(const std::vector<T, HostAlloc>& vec);

        template <concepts::InputIter<T> InputIt>
        [[nodiscard]] inline static DeviceArr
        fromIter(InputIt begin, InputIt end);

        // Fancy constructors
        [[nodiscard]] static DeviceArr<T, AllocatorTemplate, AllocatorParams...>
        createFull(std::size_t nElems, T fillval);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate, AllocatorParams...>
        createOnes(std::size_t nElems);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate, AllocatorParams...>
        createZeros(std::size_t nElems);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate, AllocatorParams...>
        createSequence(T stop);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate, AllocatorParams...>
        createSequence(T start, T stop, T step = 1);
    };

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(const cls& other)
    : DeviceArr(other.size()) {}

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(cls&& other) {
        mpMemLife.swap(other.mpMemLife);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(
        std::unique_ptr<mem_life_t>&& pMemLife)
    : mpMemLife{std::move(pMemLife)} {}

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(
        const size_type size)
    : DeviceArr::DeviceArr(size, T()) {}

    DEFINE_CUDA_ERROR(
        DeviceArrFillError,
        "Could not fill memory with initial value")

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(const size_type size, const T fillval)
    : DeviceArr{std::make_unique<mem_life_t>(size)} {
        fill(fillval);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(const size_type size, const T* const fillval)
    : DeviceArr(size, *fillval) {}

    DEFINE_CUDA_ERROR(
        DeviceArrToDeviceError,
        "Could not copy memory from host to device")

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(const host_ptr<T> arr, const size_type size)
    : DeviceArr(size) {
        const auto status = cudaMemcpy(
            mpMemLife->data(), arr, sizeBytes(), cudaMemcpyHostToDevice);

        DeviceArrToDeviceError::check(status);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(std::initializer_list<T> iniList)
    : DeviceArr(iniList.begin(), iniList.end()) {}

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <std::input_iterator InputIt>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(InputIt begin, InputIt end)
    : DeviceArr(std::vector<T>(begin, end)) {}

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <typename HostAlloc>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(const std::vector<T, HostAlloc>& vec)
    : DeviceArr(const_cast<host_ptr<T>>(vec.data()), vec.size()) {}

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <std::size_t N>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::DeviceArr(const std::array<T, N>& arr)
    : DeviceArr(const_cast<host_ptr<T>>(arr.data()), N) {}

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::size_type
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::size() const noexcept {
        return mpMemLife->size();
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>::size_type
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::sizeBytes() const noexcept {
        return mpMemLife->sizeBytes();
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline device_ptr<T>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::data() const noexcept {
        return mpMemLife->data();
    }

    DEFINE_CUDA_ERROR(
        DeviceArrCopyError,
        "Could not copy memory from device to device")

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <DeviceArrValue U>
    inline DeviceArr<U, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::copy() const {
        DeviceArr<U, AllocatorTemplate> newArr(size());
        try {
            launcher::copy(newArr.data(), data(), size());
        } catch (const launcher::DeviceCopyError& e) {
            DeviceArrCopyError::check(e.errCode);
        }

        return newArr;
    }

    DEFINE_CUDA_ERROR(
        DeviceArrToHostError,
        "Could not copy memory from device to host")

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline void
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::fill(const T fillval) {
        try {
            launcher::fill(mpMemLife->data(), size(), fillval);
            cuda_utils::host::checkKernelLaunch("");
        } catch (const cuda_utils::host::DeviceKernelLaunchError& e) {
            throw DeviceArrFillError(e.errCode);
        }
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <DeviceArrValue U, concepts::MappingFn<T, U> F>
    inline DeviceArr<U, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::transform(F f) const {
        auto newArr = this->copy<U>();
        newArr.transform_inplace(f);

        return newArr;
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <concepts::MappingFn<T> F>
    inline void
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::transform_inplace(F f) {
        launcher::transform(data(), data(), size(), f);
        cuda_utils::host::checkKernelLaunch("transform");
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <launcher::Reductible Acc, concepts::Reductor<T, Acc> F>
    inline Acc
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::reduce(F f) const {
        return launcher::reduce(
            data(), size(), f, launcher::ReduceStrategy::RECURSE);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline std::string
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::toString() const {
        return utils::fmtVec(toHost());
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline void
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::print(const std::string& end, std::ostream& out) const {
        utils::printVec(toHost(), end, out);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <typename HostAlloc>
    inline std::vector<T, HostAlloc>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::toHost() const {
        // zeroed-out memory
        std::vector<T, HostAlloc> hostVec(size());

        const auto status = cudaMemcpy(
            hostVec.data(), mpMemLife->data(), sizeBytes(), cudaMemcpyDeviceToHost);

        DeviceArrToHostError::check(status);

        return hostVec;
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::fromDevicePtr(
        device_ptr<T>&& dp, size_type size) {

        const auto newMemLife = std::make_unique<mem_life_t>(dp, size);

        return DeviceArr(std::move(newMemLife));
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <std::size_t N>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::fromArray(const std::array<T, N>& arr) {
        return {arr};
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::fromRawArray(const T* const arr, std::size_t size) {
        return {arr, size};
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <typename HostAlloc>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::fromVector(const std::vector<T, HostAlloc>& vec) {
        return {vec};
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    template <concepts::InputIter<T> InputIt>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::fromIter(InputIt begin, InputIt end) {
        return {begin, end};
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::createFull(const std::size_t nElems, const T fillval) {
        DeviceArr<T, AllocatorTemplate, AllocatorParams...> arr(nElems);
        arr.fill(fillval);

        return arr;
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::createOnes(const std::size_t nElems) {
        return createFull(nElems, static_cast<T>(1));
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::createZeros(const std::size_t nElems) {
        const T zero = T();
        return createFull(nElems, zero);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::createSequence(const T stop) {
        return createSequence(0, stop);
    }

    template <
        DeviceArrValue T,

        template <typename TT, typename... PP>
        typename AllocatorTemplate,

        typename... AllocatorParams>
    inline DeviceArr<T, AllocatorTemplate, AllocatorParams...>
    DeviceArr<T, AllocatorTemplate, AllocatorParams...>::createSequence(const T start, const T stop, const T step) {

        const auto        delta = (start < stop) ? (stop - start) : (start - stop);
        const std::size_t nElems =
            static_cast<std::size_t>(delta) / static_cast<std::size_t>(step);

        DeviceArr<T, AllocatorTemplate, AllocatorParams...> counted(nElems);

        launcher::counting(counted.data(), start, stop, step);
    }
} // namespace raii
