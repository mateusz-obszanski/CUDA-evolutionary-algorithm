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
        template <typename TT> typename AllocatorTemplate = allocator::DeviceAllocatorD1>
    class DeviceArr {
    private:
        DeviceArr();

    protected:
        using mem_life_t     = life::DeviceMemoryLife<T, AllocatorTemplate<T>>;
        using allocator_type = mem_life_t::allocator_type;

        // This is a unique_ptr to avoid double free when destructor is called
        // after std::move
        std::unique_ptr<mem_life_t> mpMemLife;

    public:
        using cls        = DeviceArr<T, AllocatorTemplate>;
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

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        [[nodiscard]] inline DeviceArr(const std::vector<T, Alloc>& vec);

        // Getters
        [[nodiscard]] size_type
        size() const noexcept;

        [[nodiscard]] size_type
        sizeBytes() const noexcept;

        [[nodiscard]] device_ptr<T>
        data() const noexcept;

        template <DeviceArrValue U = T>
        [[nodiscard]] DeviceArr<U, AllocatorTemplate>
        copy() const;

        // Modifiers
        // cudaMemset sets bytes, so for sizeof(T) > 1 values representable
        // by >1 bytes cannot be set this way, hence need to implement fill
        void
        fill(T x);

        template <DeviceArrValue U = T, concepts::MappingFn<T, U> F>
        [[nodiscard]] DeviceArr<U, AllocatorTemplate>
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

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        [[nodiscard]] inline std::vector<T, Alloc>
        toHost() const;

        // Static methods
        template <std::size_t N>
        [[nodiscard]] inline static DeviceArr
        fromArray(const std::array<T, N>& arr);

        [[nodiscard]] static DeviceArr
        fromRawArray(const T* arr, std::size_t size);

        template <typename Alloc = std::pmr::polymorphic_allocator<T>>
        [[nodiscard]] inline static DeviceArr
        fromVector(const std::vector<T, Alloc>& vec);

        template <concepts::InputIter<T> InputIt>
        [[nodiscard]] inline static DeviceArr
        fromIter(InputIt begin, InputIt end);

        // Fancy constructors
        [[nodiscard]] static DeviceArr<T, AllocatorTemplate>
        createFull(std::size_t nElems, T fillval);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate>
        createOnes(std::size_t nElems);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate>
        createZeros(std::size_t nElems);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate>
        createSequence(T stop);

        [[nodiscard]] static DeviceArr<T, AllocatorTemplate>
        createSequence(T start, T stop, T step = 1);
    };

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const cls& other)
    : DeviceArr(other.size()) {}

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(cls&& other) {
        mpMemLife.swap(other.mpMemLife);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(
        std::unique_ptr<mem_life_t>&& pMemLife)
    : mpMemLife{std::move(pMemLife)} {}

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const size_type size)
    : DeviceArr::DeviceArr(size, T{}) {}

    DEFINE_CUDA_ERROR(
        DeviceArrFillError,
        "Could not fill memory with initial value")

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const size_type size, const T fillval)
    : DeviceArr{std::make_unique<mem_life_t>(size)} {
        fill(fillval);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const size_type size, const T* const fillval)
    : DeviceArr(size, *fillval) {}

    DEFINE_CUDA_ERROR(
        DeviceArrToDeviceError,
        "Could not copy memory from host to device")

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const device_ptr<T> arr, const size_type size)
    : DeviceArr(size) {
        const auto status = cudaMemcpy(
            mpMemLife->data(), arr, sizeBytes(), cudaMemcpyHostToDevice);

        DeviceArrToDeviceError::check(status);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(std::initializer_list<T> iniList)
    : DeviceArr(iniList.begin(), iniList.end()) {}

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <std::input_iterator InputIt>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(InputIt begin, InputIt end)
    : DeviceArr(std::vector<T>(begin, end)) {}

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <typename Alloc>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const std::vector<T, Alloc>& vec)
    : DeviceArr(const_cast<host_ptr<T>>(vec.data()), vec.size()) {}

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <std::size_t N>
    inline DeviceArr<T, AllocatorTemplate>::DeviceArr(const std::array<T, N>& arr)
    : DeviceArr(const_cast<host_ptr<T>>(arr.data()), N) {}

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::size_type
    DeviceArr<T, AllocatorTemplate>::size() const noexcept {
        return mpMemLife->size();
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>::size_type
    DeviceArr<T, AllocatorTemplate>::sizeBytes() const noexcept {
        return mpMemLife->sizeBytes();
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline device_ptr<T>
    DeviceArr<T, AllocatorTemplate>::data() const noexcept {
        return mpMemLife->data();
    }

    DEFINE_CUDA_ERROR(
        DeviceArrCopyError,
        "Could not copy memory from device to device")

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <DeviceArrValue U>
    inline DeviceArr<U, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::copy() const {
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

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline void
    DeviceArr<T, AllocatorTemplate>::fill(const T fillval) {
        try {
            launcher::fill(mpMemLife->data(), size(), fillval);
            cuda_utils::host::checkKernelLaunch("");
        } catch (const cuda_utils::host::DeviceKernelLaunchError& e) {
            throw DeviceArrFillError(e.errCode);
        }
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <DeviceArrValue U, concepts::MappingFn<T, U> F>
    inline DeviceArr<U, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::transform(F f) const {
        auto newArr = this->copy<U>();
        newArr.transform_inplace(f);

        return newArr;
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <concepts::MappingFn<T> F>
    inline void
    DeviceArr<T, AllocatorTemplate>::transform_inplace(F f) {
        launcher::transform(data(), data(), size(), f);
        cuda_utils::host::checkKernelLaunch("transform");
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <launcher::Reductible Acc, concepts::Reductor<T, Acc> F>
    inline Acc
    DeviceArr<T, AllocatorTemplate>::reduce(F f) const {
        return launcher::reduce(
            data(), size(), f, launcher::ReduceStrategy::RECURSE);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline std::string
    DeviceArr<T, AllocatorTemplate>::toString() const {
        return utils::fmtVec(toHost());
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline void
    DeviceArr<T, AllocatorTemplate>::print(const std::string& end, std::ostream& out) const {
        utils::printVec(toHost(), end, out);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <typename Alloc>
    inline std::vector<T, Alloc>
    DeviceArr<T, AllocatorTemplate>::toHost() const {
        // zeroed-out memory
        std::vector<T, Alloc> hostVec(size());

        const auto status = cudaMemcpy(
            hostVec.data(), mpMemLife->data(), sizeBytes(), cudaMemcpyDeviceToHost);

        DeviceArrToHostError::check(status);

        return hostVec;
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <std::size_t N>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::fromArray(const std::array<T, N>& arr) {
        return {arr};
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::fromRawArray(const T* const arr, std::size_t size) {
        return {arr, size};
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <typename Alloc>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::fromVector(const std::vector<T, Alloc>& vec) {
        return {vec};
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    template <concepts::InputIter<T> InputIt>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::fromIter(InputIt begin, InputIt end) {
        return {begin, end};
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::createFull(const std::size_t nElems, const T fillval) {
        DeviceArr<T, AllocatorTemplate> arr(nElems);
        arr.fill(fillval);

        return arr;
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::createOnes(const std::size_t nElems) {
        return createFull(nElems, static_cast<T>(1));
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::createZeros(const std::size_t nElems) {
        const T zero = T();
        return createFull(nElems, zero);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::createSequence(const T stop) {
        return createSequence(0, stop);
    }

    template <DeviceArrValue T, template <typename TT> typename AllocatorTemplate>
    inline DeviceArr<T, AllocatorTemplate>
    DeviceArr<T, AllocatorTemplate>::createSequence(const T start, const T stop, const T step) {

        const auto        delta = (start < stop) ? (stop - start) : (start - stop);
        const std::size_t nElems =
            static_cast<std::size_t>(delta) / static_cast<std::size_t>(step);

        DeviceArr<T, AllocatorTemplate> counted(nElems);

        launcher::counting(counted.data(), start, stop, step);
    }
} // namespace raii
