#pragma once

#include "../common_concepts.hxx"
#include "./kernel_utils.h"
#include "./types.h"
#include <concepts>

namespace device {
namespace combining {

namespace kernel {

template <typename X, typename Y, typename Out1, typename Out2,
          std::convertible_to<bool> MaskT>

    requires ConvertibleToAll<X, Out1, Out2> and ConvertibleToAll<Y, Out1, Out2>
__global__ void
crossover(device_ptr_in<X> xs, const std::size_t n, device_ptr_in<Y> ys,
          device_ptr_in<MaskT> mask, device_ptr_out<Out1> out1,
          device_ptr_out<Out2> out2) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n)
        return;

    const auto x = xs[tid];
    const auto y = ys[tid];

    if (mask[tid]) {
        out1[tid] = x;
        out2[tid] = y;
    } else {
        out1[tid] = y;
        out2[tid] = x;
    }
}

template <typename X, typename Y, typename Out1, typename Out2,
          typename Stencil, UnaryPredicate<Stencil> Pred>

    requires ConvertibleToAll<X, Out1, Out2> and ConvertibleToAll<Y, Out1, Out2>
__global__ void
crossover(device_ptr_in<X> xs, const std::size_t n, device_ptr_in<Y> ys,
          device_ptr_in<Stencil> stencil, Pred p, device_ptr_out<Out1> out1,
          device_ptr_out<Out2> out2) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n)
        return;

    const auto x = xs[tid];
    const auto y = ys[tid];

    if (p(stencil[tid])) {
        out1[tid] = x;
        out2[tid] = y;
    } else {
        out1[tid] = y;
        out2[tid] = x;
    }
}

} // namespace kernel

template <typename IterX, typename IterY, typename IterOut1, typename IterOut2>
concept Crossoverable =
    IterValConvertibleToIterVal<IterX, IterOut1, IterOut2> and
    IterValConvertibleTo<IterY, IterOut1, IterOut2>;

template <typename IterX, typename IterY, typename IterOut1, typename IterOut2,
          IterValConvertibleTo<bool> IterMask>

    requires Crossoverable<IterX, IterY, IterOut1, IterOut2>

inline void
crossover(IterX beginX, IterX endX, IterY beginY, IterMask beginMask,
          IterOut1 beginOut1, IterOut2 beginOut2,
          const cudaStream_t stream = 0) {

    using device::kernel::utils::iterToRawPtr;

    const auto n = thrust::distance(beginX, endX);

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(n);

    kernel::crossover<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT, 0,
                        stream>>>(
        iterToRawPtr(beginX), n, iterToRawPtr(beginY), iterToRawPtr(beginMask),
        iterToRawPtr(beginOut1), iterToRawPtr(beginOut2));
}

template <
    typename IterX, typename IterY, typename IterOut1, typename IterOut2,
    typename IterStencil,
    UnaryPredicate<typename std::iterator_traits<IterStencil>::value_type> Pred>

    requires Crossoverable<IterX, IterY, IterOut1, IterOut2>

inline void
crossover(IterX beginX, IterX endX, IterY beginY, IterStencil beginStencil,
          Pred p, IterOut1 beginOut1, IterOut2 beginOut2,
          const cudaStream_t stream = 0) {

    using device::kernel::utils::iterToRawPtr;

    const auto n = thrust::distance(beginX, endX);

    const auto nBlocks = device::kernel::utils::calcBlockNum1D(n);

    kernel::crossover<<<nBlocks, device::kernel::utils::BLOCK_SIZE_DEFAULT, 0,
                        stream>>>(iterToRawPtr(beginX), n, iterToRawPtr(beginY),
                                  iterToRawPtr(beginStencil), p,
                                  iterToRawPtr(beginOut1),
                                  iterToRawPtr(beginOut2));
}

} // namespace combining
} // namespace device
