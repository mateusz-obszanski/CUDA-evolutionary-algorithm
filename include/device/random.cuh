#pragma once

#include "../types/concepts.hxx"
#include "../types/types.hxx"
#include "./errors.cuh"
#include "./memory/allocator.cuh"
#include <concepts>
#include <cuda/std/bit>
#include <curand_kernel.h>
#include <iterator>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <type_traits>
#include <vector>

namespace device {
namespace random {

using seed_t     = ullong;
using sequence_t = ullong;
using offset_t   = ullong;

template <typename State>
concept IsInitializableRndState =
    types::concepts::AnyOf<
        State, curandStateXORWOW_t, curandStateMRG32k3a_t, curandStatePhilox4_32_10_t> and
    device::memory::raii::DeviceStorable<State>;

template <
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Alloc = device::memory::allocator::DeviceAllocator>

    requires device::memory::allocator::concepts::DeviceAllocator<Alloc<State>>
using RndStateMemory = device::memory::raii::Memory<State, Alloc>;

struct RndStateInitParams {
    const seed_t     seed      = 0;
    const sequence_t first_seq = 0;
    const offset_t   offset    = 0;
};

namespace kernel {

template <IsInitializableRndState State = curandState>
__global__ void
initialize_rnd_states(
    State* const states, std::size_t length,
    const RndStateInitParams params = RndStateInitParams{}) {

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= length)
        return;

    // Each thread gets same seed, a different sequence number
    curand_init(params.seed, params.first_seq + tid, params.offset, &states[tid]);
}

} // namespace kernel

template <IsInitializableRndState State = curandState>
inline void
initialize_rnd_states(
    State* const states, std::size_t length,
    const RndStateInitParams& params = RndStateInitParams{}, const cudaStream_t& stream = 0) {

    using namespace device::kernel::utils;

    const auto nBlocks = calcBlockNum1D(length);
    kernel::initialize_rnd_states<<<nBlocks, BLOCK_SIZE_DEFAULT, 0, stream>>>(states, length, params);
    errors::check();
}

template <
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>
inline void
initialize_rnd_states(
    RndStateMemory<State, Allocator>& states,
    const RndStateInitParams& params = RndStateInitParams{}, const cudaStream_t& stream = 0) {

    initialize_rnd_states<State>(states.data(), states.size(), params, stream);
}

namespace functional {

template <typename F, typename State>
concept RndDistributionFunctor = IsInitializableRndState<State> and
                                 requires(F f, State s) {{f(s)} -> std::same_as<float>; };

template <IsInitializableRndState State = curandState>
struct UniformGen {
    __device__ float
    operator()(State& state) const {
        return curand_uniform(&state);
    }
};

} // namespace functional

namespace utils {

template <
    typename IteratorOut,
    IsInitializableRndState State,

    template <typename TT>
    typename Allocator,

    functional::RndDistributionFunctor<State> G>
inline void
generate_distribution(
    IteratorOut begin, IteratorOut end,
    RndStateMemory<State, Allocator>& states, const cudaStream_t& stream) {

    thrust::transform(
        thrust::device.on(stream),
        states.begin_thrust(),
        states.begin_thrust() + device::iterator::distance(begin, end),
        begin,
        G{});
}

} // namespace utils

template <
    typename IteratorOut,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>
void
uniform(
    IteratorOut begin, IteratorOut end,
    RndStateMemory<State, Allocator>& states, const cudaStream_t& stream = 0) {

    utils::generate_distribution<IteratorOut, State, Allocator, functional::UniformGen<State>>(
        begin, end, states, stream);
}

} // namespace random
} // namespace device
