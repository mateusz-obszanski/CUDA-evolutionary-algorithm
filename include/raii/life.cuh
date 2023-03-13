#pragma once

#include "./allocators.cuh"

namespace raii {
    namespace life {
        template <
            typename T,
            typename Allocator = allocator::DeviceAllocatorD1<T>>
        class DeviceMemoryLife {
        public:
            using allocator_type = Allocator;
            using value_t        = allocator_type::value_type;
            using pointer        = allocator_type::pointer;
            using size_type      = allocator_type::size_type;

            [[nodiscard]] DeviceMemoryLife(const DeviceMemoryLife<T, Allocator>&);
            [[nodiscard]] DeviceMemoryLife(size_type size);
            ~DeviceMemoryLife();

            template <typename U = T>
            [[nodiscard]] inline device_ptr<U>
            data() const noexcept;

            [[nodiscard]] size_type
            size() const noexcept;

            [[nodiscard]] size_type
            sizeBytes() const noexcept;

        private:
            [[nodiscard]] DeviceMemoryLife();
            [[nodiscard]] DeviceMemoryLife(size_type size, pointer&& pData);

        protected:
            const size_type mSize  = 0;
            pointer         mpData = nullptr;

            Allocator mAllocator;
        };

        template <typename T, typename Allocator>
        DeviceMemoryLife<T, Allocator>::DeviceMemoryLife(
            const DeviceMemoryLife<T, Allocator>& other)
        : DeviceMemoryLife(other.mSize) {}

        /// @brief Assumes that pData has already been allocated on a device
        /// @tparam T
        /// @param size
        /// @param pData
        template <typename T, typename Allocator>
        inline DeviceMemoryLife<T, Allocator>::DeviceMemoryLife(
            const size_type size, pointer&& pData)
        : mSize(size), mpData(std::move(pData)) {}

        template <typename T, typename Allocator>
        inline DeviceMemoryLife<T, Allocator>::DeviceMemoryLife(
            const size_type size)
        : mSize(size), mAllocator() {
            mpData = mAllocator.allocate(sizeBytes());
        }

        template <typename T, typename Allocator>
        inline DeviceMemoryLife<T, Allocator>::~DeviceMemoryLife() {
            try {
                mAllocator.deallocate(mpData, sizeBytes());
                // no throwing inside the destructor
            } catch (const allocator::DeviceFreeError&) {
                std::cerr << "\nERROR: could not deallocate device memory\n";
            } catch (...) {
                std::cerr
                    << "\nUNKNOWN ERROR: could not deallocate device memory\n";
            }
        }

        template <typename T, typename Allocator>
        template <typename U>
        inline device_ptr<U>
            DeviceMemoryLife<T, Allocator>::template data<U>() const noexcept {
            return mpData;
        }

        template <typename T, typename Allocator>
        inline DeviceMemoryLife<T, Allocator>::size_type
        DeviceMemoryLife<T, Allocator>::size() const noexcept {
            return mSize;
        }

        template <typename T, typename Allocator>
        inline DeviceMemoryLife<T, Allocator>::size_type
        DeviceMemoryLife<T, Allocator>::sizeBytes() const noexcept {
            return mSize * sizeof(T);
        }
    } // namespace life
} // namespace raii
