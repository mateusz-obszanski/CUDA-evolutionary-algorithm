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
    };

namespace cuda_utils {
    namespace host {
        class CudaError : public std::exception {
            cudaError_t err;

        protected:
            std::string msg;

        public:
            CudaError(cudaError_t err)
            : err{err}, msg{std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err)} {}

            const char* what() const noexcept override {
                return msg.c_str();
            }
        };
    } // namespace host
} // namespace cuda_utils
