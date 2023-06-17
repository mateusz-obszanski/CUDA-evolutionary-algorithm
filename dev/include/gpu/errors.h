#pragma once

#include "./types.h"
#include <cuda_runtime.h>
#include <string>

namespace device {
namespace errors {

class DeviceOperationFailedError : public std::exception {
public:
    const cudaError_t errCode;
    const std::string msg;

    DeviceOperationFailedError(cudaError_t errCode, int lineNum) noexcept
    : errCode{errCode},
      msg{std::to_string(lineNum) + " " + cudaGetErrorName(errCode) + ": " +
          cudaGetErrorString(errCode)} {}

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

void
check(const cudaError_t status = cudaGetLastError(), const int lineNum = -1) {

    if (status != cudaSuccess)
        throw DeviceOperationFailedError(status, lineNum);
}

} // namespace errors
} // namespace device
