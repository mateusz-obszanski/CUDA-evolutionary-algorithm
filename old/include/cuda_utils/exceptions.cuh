#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define DEFINE_CUDA_ERROR(ERR_NAME, MSG)                      \
    class ERR_NAME : public ::cuda_utils::host::DeviceError { \
    public:                                                   \
        ERR_NAME(cudaError_t status)                          \
        : ::cuda_utils::host::DeviceError{status} {           \
            /* prepend full message with error class name */  \
            this->fullMsg = #ERR_NAME ": " + this->fullMsg;   \
        }                                                     \
        static inline void                                    \
        check(cudaError_t status) {                           \
            if (status != cudaSuccess)                        \
                throw ERR_NAME(status);                       \
        }                                                     \
        static inline void                                    \
        check() {                                             \
            check(cudaGetLastError());                        \
        }                                                     \
    };

namespace cuda_utils {
    namespace host {
        class DeviceError : public std::exception {
        public:
            const cudaError_t errCode;
            const std::string errName;
            const std::string errString;

            DeviceError(cudaError_t errCode)
            : errCode{errCode},
              errName{cudaGetErrorName(errCode)},
              errString{cudaGetErrorString(errCode)},
              fullMsg{errName + ", reason: " + errString} {}

            const char*
            what() const noexcept override {
                return fullMsg.c_str();
            }

        protected:
            std::string fullMsg;
        };

        class DeviceKernelLaunchError : public DeviceError {
        public:
            DeviceKernelLaunchError(const std::string kernelName,
                                    const cudaError_t errCode)
            : DeviceError{errCode} {
                this->fullMsg =
                    "DeviceKernelLaunchError: kernel " + kernelName + fullMsg;
            }
        };

        /// @brief raises cuda_utils::host::DeviceKernelLaunchError
        /// @param kernelName
        void
        checkKernelLaunch(const std::string& kernelName) {
            // launch parameters error
            if (const auto status = cudaPeekAtLastError())
                throw DeviceKernelLaunchError(kernelName, status);

            // execution error

            // can be ommited if there is subsequent device operation, e.g.
            // cudaMemcpy - this next operation will return either previous
            // kernel's error or its own
            if (const auto status = cudaDeviceSynchronize())
                throw DeviceKernelLaunchError(kernelName, status);
        }
    } // namespace host
} // namespace cuda_utils
