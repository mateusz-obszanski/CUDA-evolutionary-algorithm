#pragma once

#include "../types/concepts.hxx"
#include "../types/types.hxx"
#include "./errors.cuh"
#include "./memory/allocator.cuh"
#include <concepts>
#include <cuda/std/bit>
#include <curand_kernel.h>
#include <iterator>
#include <thrust/device_ptr.h>
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
inline void
uniform(
    IteratorOut begin, IteratorOut end,
    RndStateMemory<State, Allocator>& states, const cudaStream_t& stream = 0) {

    utils::generate_distribution<IteratorOut, State, Allocator, functional::UniformGen<State>>(
        begin, end, states, stream);
}

template <typename IterIn, typename IterOut, typename ThreshT>
    requires std::constructible_from<typename IterOut::value_type, bool> and
             types::concepts::GtComparableWith<typename IterIn::value_type, ThreshT>
inline void
threshold(IterIn begin, IterIn end, IterOut out, const ThreshT thresh) {
    thrust::transform(
        begin, end, out, [=] __device__(const float p) -> bool { return p > thresh; });
}

template <
    typename IterOut,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>

    requires std::constructible_from<typename IterOut::value_type, bool>
inline void
mask(
    IterOut begin, IterOut end,
    RndStateMemory<State, Allocator>& states, const float maskProbability = 0.5f,
    const cudaStream_t& stream = 0) {

    using Probs    = thrust::device_vector<float>;
    using ProbIter = Probs::iterator;

    ProbIter probsBegin, probsEnd;
    Probs    probs;

    if constexpr (std::same_as<typename IterOut::value_type, float>) {
        // do not allocate additional memory, use provided iterators
        probsBegin = thrust::device_pointer_cast(begin);
        probsEnd   = thrust::device_pointer_cast(end);
    } else {
        const auto n = device::iterator::distance(begin, end);
        probs        = Probs(n);

        probsBegin = probs.begin();
        probsEnd   = probs.end();
    }

    uniform(probsBegin, probsEnd, states, stream);
    threshold(probsBegin, probsEnd, begin, maskProbability);
}

} // namespace random
} // namespace device
