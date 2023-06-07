#pragma once

#include "../text.hxx"
#include <cuda_runtime.h>
#include <experimental/source_location> // this has been merged to c++20, but does not work with nvcc
#include <stdexcept>

namespace device {
namespace errors {

namespace {

using std::experimental::source_location;

}

class DeviceOperationFailedError : public std::exception {
public:
    const cudaError_t     errCode;
    const std::string     msg;
    const source_location lineInfo;

    DeviceOperationFailedError(cudaError_t errCode, source_location lineInfo) noexcept
    : errCode{errCode},
      lineInfo{lineInfo},
      msg{
          text::fmtLineInfo(lineInfo) + " " +
          cudaGetErrorName(errCode) + ": " +
          cudaGetErrorString(errCode)} {}

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

void
check(
    const cudaError_t      status   = cudaGetLastError(),
    const source_location& lineInfo = source_location::current()) {

    if (status != cudaSuccess)
        throw DeviceOperationFailedError(status, lineInfo);
}

} // namespace errors
} // namespace device
