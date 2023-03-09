#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define DEFINE_CUDA_ERROR(ERR_NAME, MSG)                                       \
    class ERR_NAME : public ::cuda_utils::host::CudaError {                    \
    public:                                                                    \
        ERR_NAME(cudaError_t status) : ::cuda_utils::host::CudaError{status} { \
            this->msg = #ERR_NAME ": " + this->msg;                            \
        }                                                                      \
        static inline void check(cudaError_t status) {                         \
            if (status != cudaSuccess)                                         \
                throw ERR_NAME(status);                                        \
        }                                                                      \
        static inline void check() {                                           \
            check(cudaGetLastError());                                         \
        }                                                                      \
    };

namespace cuda_utils {
    namespace host {
        class CudaError : public std::exception {
        public:
            const cudaError_t err;

            CudaError(cudaError_t err)
            : err{err}, msg{std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err)} {}

            const char* what() const noexcept override {
                return msg.c_str();
            }

        protected:
            std::string msg;
        };

        class CudaKernelLaunchError : public CudaError {
        public:
            CudaKernelLaunchError(const std::string kernelName, const cudaError_t err) : CudaError{err} {
                this->msg = "CudaKernelLaunchError: kernel " + kernelName + ", reason: " + std::string(cudaGetErrorName(err)) + " - " + std::string(cudaGetErrorString(err));
            }
        };

        void checkKernelLaunch(const std::string& kernelName) {
            if (const auto status = cudaDeviceSynchronize())
                throw CudaKernelLaunchError(kernelName, status);
        }
    } // namespace host
} // namespace cuda_utils
