#pragma once

#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../cuda_utils/exceptions.cuh"
#include "../kernels/kernels.cuh"
#include "../types.hxx"
#include <concepts>
#include <cstddef>

namespace launcher {
    namespace utils {
        template <typename Size = std::size_t>
        PairOf<Size>
        calcLaunchParams(const Size nElems) {
            constexpr std::size_t blockSize = 1024;
            const auto            nBlocks   = cuda_utils::divCeil<std::size_t>(nElems, blockSize);

            return {nBlocks, blockSize};
        }
    } // namespace utils

    template <typename T>
    inline void
    fill(dRawVecOut<T> arr, const std::size_t nElems, const T fillval) {
        const auto&& [nBlocks, blockSize] = utils::calcLaunchParams(nElems);

        kernel::fill<<<nBlocks, blockSize>>>(arr, nElems, fillval);
        cuda_utils::host::checkKernelLaunch("fill");
    }

    template <typename T>
    inline void
    fill(
        dRawVecOut<T> arr, const std::size_t nElems, const T fillval,
        const std::size_t begin) {

        fill(arr + begin, nElems - begin, fillval);
    }

    template <typename T>
    inline void
    fill(
        dRawVecOut<T> arr, const std::size_t nElems, const T fillval,
        const std::size_t begin, const std::size_t end) {

        fill(arr + begin, nElems - begin - end, fillval);
    }

    DEFINE_CUDA_ERROR(DeviceCopyError, "Could not copy data from device to device")

    template <typename T>
    inline void
    copy(dRawVecOut<T> dest, dRawVecIn<T> src, std::size_t nElems) {
        const auto status = cudaMemcpy(dest, src, sizeof(T) * nElems, cudaMemcpyDeviceToDevice);
        DeviceCopyError::check(status);
    }

    template <typename SrcT, concepts::ConstructibleButDifferent<SrcT> DestT>
    inline void
    copy(dRawVecOut<DestT> dest, dRawVecIn<SrcT> src, std::size_t nElems) {
        const auto&& [nBlocks, blockSize] = utils::calcLaunchParams(nElems);

        kernel::copy<<<nBlocks, blockSize>>>(dest, src, nElems);

        if (const auto status = cudaDeviceSynchronize())
            throw DeviceCopyError(status);
    }
} // namespace launcher
