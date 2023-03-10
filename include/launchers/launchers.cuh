#pragma once

#include "../concepts.hxx"
#include "../cuda_types.cuh"
#include "../cuda_utils/cuda_utils.cuh"
#include "../cuda_utils/exceptions.cuh"
#include "../kernels/kernels.cuh"
#include "../types.hxx"
#include "./_utils.cuh"
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace launcher {
    enum class FillMode {
        ALLOW_MEMSET_SHORT_OR_ZERO,
        ALWAYS_USE_KERNEL,
    };

    namespace {
        template <std::integral T>
        inline constexpr bool
        shorterThanByteOrZero(const T val) {
            return val == 0;
        }

        template <typename T>
            requires std::same_as<T, short> or std::same_as<T, unsigned short>
        inline constexpr bool
        shorterThanByteOrZero(const T val) {
            return true;
        }

        template <typename T>
        inline constexpr bool
        shorterThanByteOrZero(const T val) {
            return false;
        }
    } // namespace

    DEFINE_CUDA_ERROR(FillError, "Could not fill device memory")

    template <typename T>
    inline void
    fill(
        dRawVecOut<T>     arr,
        const std::size_t nElems, const T fillval,
        FillMode mode = FillMode::ALLOW_MEMSET_SHORT_OR_ZERO) {

        if (std::is_integral_v<T> and shorterThanByteOrZero(fillval) and mode == FillMode::ALLOW_MEMSET_SHORT_OR_ZERO)
            if (const auto err = cudaMemset(arr, fillval, nElems * sizeof(T)))
                throw FillError(err);

        const auto nBlocks = utils::calcBlockNum1D(nElems);

        kernel::fill<<<nBlocks, utils::BLOCK_SIZE_1D>>>(arr, nElems, fillval);
        try {
            cuda_utils::host::checkKernelLaunch("fill");
        } catch (const cuda_utils::host::CudaKernelLaunchError& err) {
            throw FillError(err.err);
        }
    }

    template <typename T>
    inline void
    fill(
        dRawVecOut<T> arr, const std::size_t nElems, const T fillval,
        const std::size_t begin, FillMode mode = FillMode::ALLOW_MEMSET_SHORT_OR_ZERO) {

        fill(arr + begin, nElems - begin, fillval, mode);
    }

    template <typename T>
    inline void
    fill(
        dRawVecOut<T> arr, const std::size_t nElems, const T fillval,
        const std::size_t begin, const std::size_t end, FillMode mode = FillMode::ALLOW_MEMSET_SHORT_OR_ZERO) {

        fill(arr + begin, nElems - begin - end, fillval, mode);
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
        const auto nBlocks = utils::calcBlockNum1D(nElems);

        kernel::copy<<<nBlocks, utils::BLOCK_SIZE_1D>>>(dest, src, nElems);

        if (const auto status = cudaDeviceSynchronize())
            throw DeviceCopyError(status);
    }

    using kernel::CountingAble;

    // TODO launcher/kernel counting
    template <CountingAble T>
    inline void
    counting(dRawVecOut<T> out, const std::size_t nElems, const T start, const T step) {
        const auto nBlocks = utils::calcBlockNum1D(nElems);

        kernel::counting<<<nBlocks, utils::BLOCK_SIZE_1D>>>(out, nElems, start, step);
        cuda_utils::host::checkKernelLaunch("counting");
    }
} // namespace launcher
