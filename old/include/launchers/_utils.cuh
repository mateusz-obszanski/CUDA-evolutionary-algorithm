#pragma once

#include "../macros.hxx"
#include "./cuda_utils/cuda_utils.cuh"
#include <concepts>
#include <cstddef>
#include <iostream>

namespace launcher {
    namespace utils {
        constexpr std::size_t BLOCK_SIZE_1D = 1024;

        template <typename Size = std::size_t>
        constexpr Size
        calcBlockNum1D(const Size nElems) {
            return cuda_utils::divCeil<std::size_t>(nElems, BLOCK_SIZE_1D);
        }

        DEFINE_CUDA_ERROR(
            SimpleDeviceBufferAllocError,
            "Could not allocate memory on device")
        DEFINE_CUDA_ERROR(
            SimpleDeviceBufferToHostError,
            "Could copy data from device to host")

        /// @brief Simple RAII buffer
        template <std::default_initializable T>
        class SimpleDeviceBuffer {
        private:
            T*                mpData;
            const std::size_t size;
            const std::size_t sizeBytes;

        public:
            SimpleDeviceBuffer(
                const std::size_t size,
                const size_t      elemBytes = sizeof(T))
            : size{size}, sizeBytes{size * elemBytes} {
                const auto status = cudaMalloc<T>(&mpData, sizeBytes);

                SimpleDeviceBufferAllocError::check(status);
            }
            ~SimpleDeviceBuffer() {
                if (const auto err = cudaFree(mpData))
                    std::cerr
                        << "\nCould not free SimpleDeviceBuffer device data!\n";
            }

            T*
            data() const {
                return mpData;
            }

            std::vector<T>
            toHost() const {
                std::vector<T> hBuffer(size);

                const auto status = cudaMemcpy(
                    hBuffer.data(), mpData, sizeBytes, cudaMemcpyDeviceToHost);

                SimpleDeviceBufferToHostError::check(status);

                return hBuffer;
            }
        };
    } // namespace utils
} // namespace launcher
