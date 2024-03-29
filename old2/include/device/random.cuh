#pragma once

#include "../types/concepts.hxx"
#include "../types/types.hxx"
#include "./combining.cuh"
#include "./errors.cuh"
#include "./mask.cuh"
#include "./memory/allocator.cuh"
#include "./numeric.cuh"
#include "./reordering.cuh"
#include <concepts>
#include <cuda/std/bit>
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
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
        State,
        curandStateXORWOW_t,
        curandStateMRG32k3a_t,
        curandStatePhilox4_32_10_t> and
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
    device_ptr_inout<State> states, const uint length,
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
    const RndStateInitParams& params = RndStateInitParams{},
    const cudaStream_t&       stream = 0) {

    using namespace device::kernel::utils;

    const auto nBlocks = calcBlockNum1D(length);
    kernel::initialize_rnd_states<<<nBlocks, BLOCK_SIZE_DEFAULT, 0, stream>>>(
        states, length, params);
    errors::check();
}

template <
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>
inline void
initialize_rnd_states(
    RndStateMemory<State, Allocator>& states,
    const RndStateInitParams&         params = RndStateInitParams{},
    const cudaStream_t&               stream = 0) {

    initialize_rnd_states<State>(states.data(), states.size(), params, stream);
}

namespace functional {

template <typename F, typename State>
concept RndDistributionFunctor =
    IsInitializableRndState<State> and
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

    utils::generate_distribution<
        IteratorOut, State, Allocator, functional::UniformGen<State>>(
        begin, end, states, stream);
}

template <
    typename IterIn,
    typename IterOut,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>

    requires std::copyable<typename IterIn::value_type> and
             std::convertible_to<typename IterIn::value_type, typename IterOut::value_type>
inline IterOut
choose_with_prob_without_replacement(
    IterIn begin, IterIn end, IterOut out,
    RndStateMemory<State, Allocator>& states,
    const float                       p,
    const cudaStream_t                stream = 0) {

    const auto n = thrust::distance(begin, end);

    thrust::device_vector<float> choiceChances(n);
    uniform(choiceChances.begin(), choiceChances.end(), states, stream);

    return thrust::copy_if(
        begin, end, choiceChances.begin(), out,
        [=] __device__(const float pk) { return pk < p; });
}

template <
    typename IterOut,
    std::integral           K     = std::size_t,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>
inline void
choose_k_without_replacement(
    const std::size_t n, IterOut out, K k,
    RndStateMemory<State, Allocator>& states, const cudaStream_t stream = 0) {

    if (k <= 0 || n <= 0)
        return;

    // instead cudaMallocAsync?
    thrust::device_vector<float> priorities(n);
    uniform(priorities.begin(), priorities.end(), states, stream);

    using Idx = std::size_t;

    const thrust::counting_iterator<Idx> firstIdx;

    thrust::device_vector<Idx> indices(firstIdx, firstIdx + n);

    // stable for determinism (would be irrelevant if there was no key)
    thrust::stable_sort_by_key(
        thrust::device.on(stream),
        priorities.begin(), priorities.end(),
        indices.begin());

    thrust::copy_n(thrust::device.on(stream), indices.begin(), k, out);
}

template <
    typename IterIn,
    typename IterOut,
    std::integral           K     = std::size_t,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>

    requires ::types::concepts::ConvertibleIterVal<IterIn, IterOut>
inline void
choose_k_without_replacement(
    IterIn begin, IterIn end, IterOut out, K k,
    RndStateMemory<State, Allocator>& states, const cudaStream_t stream = 0) {

    using Idx = std::size_t;

    if (k <= 0)
        return;

    const auto n = thrust::distance(begin, end);

    thrust::device_vector<Idx> chosenIndices(k);
    choose_k_without_replacement(n, chosenIndices.begin(), k, states, stream);

    reordering::select_k(begin, out, chosenIndices.begin(), k, stream);
}

/// @brief Initializes random mask
/// @tparam IterOut
/// @tparam State
/// @param begin
/// @param end
/// @param states
/// @param maskProbability
/// @param stream
template <
    typename IterOut,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>

    requires std::constructible_from<typename IterOut::value_type, bool>
inline void
mask(
    IterOut begin, IterOut end,
    RndStateMemory<State, Allocator>& states, const float maskProbability,
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
    numeric::threshold_less(probsBegin, probsEnd, begin, maskProbability);
}

template <typename Iter, typename IterMask>
concept ShuffleMaskedAble =
    std::copyable<typename Iter::value_type> and
    std::convertible_to<typename IterMask::value_type, bool>;

template <typename Iter, typename IdxIter, typename ThrustRndEngine>
    requires std::copyable<typename Iter::value_type> and
             std::integral<typename IdxIter::value_type>
inline void
shuffle_at(
    Iter begin, IdxIter indicesBegin, IdxIter indicesEnd, ThrustRndEngine& rng) {

    using Idx    = typename IdxIter::value_type;
    using IdxVec = thrust::device_vector<Idx>;

    const auto n = thrust::distance(indicesBegin, indicesEnd);

    // make copy of indices and shuffle them, those will be destination indices
    IdxVec shuffledIndices(n);
    thrust::shuffle_copy(
        indicesBegin, indicesEnd, shuffledIndices.begin(), rng);

    reordering::swap_sparse_n(
        begin, indicesBegin, shuffledIndices.begin(), n);
}

template <typename Iter, typename IterMask, typename ThrustRndEngine>
    requires ShuffleMaskedAble<Iter, IterMask>
inline void
shuffle_masked_n(
    Iter begin, const std::size_t n, IterMask beginMask, ThrustRndEngine& rng) {

    using Idx    = std::size_t;
    using IdxVec = thrust::device_vector<Idx>;

    IdxVec maskIndices(n);
    // populating maskIndices
    const auto lastMaskIdx = mask::mask_indices_n(beginMask, n, maskIndices.begin());

    shuffle_at(begin, maskIndices.begin(), lastMaskIdx, rng);
}

template <typename Iter, typename IterMask, typename ThrustRndEngine>
    requires ShuffleMaskedAble<Iter, IterMask>
inline void
shuffle_masked(Iter begin, Iter end, IterMask beginMask, ThrustRndEngine& rng) {
    shuffle_masked_n(begin, iterator::distance(begin, end), beginMask, rng);
}

/// @brief Shuffles subset of elements pooled with probability `p`
/// @tparam Iter
/// @tparam ThrustRndEngine
/// @tparam State
/// @param begin
/// @param end
/// @param p
/// @param states
/// @param rng
template <
    typename Iter,
    typename ThrustRndEngine,
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>
inline void
shuffle_with_prob(
    Iter begin, Iter end, const float p,
    RndStateMemory<State, Allocator>& states, ThrustRndEngine& rng,
    const cudaStream_t stream = 0) {

    const auto n = thrust::distance(begin, end);

    using Idx = std::size_t;

    thrust::counting_iterator<Idx> firstIdx;
    auto                           endIdx = firstIdx + n;

    thrust::device_vector<Idx> chosenIndices(n);

    const auto endChosenIdx = choose_with_prob_without_replacement(
        firstIdx, endIdx, chosenIndices.begin(), states, p, stream);

    const auto nChosen = thrust::distance(chosenIndices.begin(), endChosenIdx);

    thrust::device_vector<Idx> targetIndices(nChosen);

    thrust::shuffle_copy(
        chosenIndices.begin(), endChosenIdx, targetIndices.begin(), rng);

    reordering::swap_sparse_n(
        begin, chosenIndices.begin(), targetIndices.begin(), nChosen, stream);
}

template <
    typename IterX,
    typename IterY,
    typename IterOut1,
    typename IterOut2,

    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Allocator = device::memory::allocator::DeviceAllocator>

    requires combining::Crossoverable<IterX, IterY, IterOut1, IterOut2>
inline void
crossover(
    IterX                             beginX,
    IterX                             endX,
    IterY                             beginY,
    IterOut1                          beginOut1,
    IterOut2                          beginOut2,
    RndStateMemory<State, Allocator>& states,
    const cudaStream_t                stream = 0) {

    const auto n = thrust::distance(beginX, endX);

    thrust::device_vector<float> swapChances(n);
    uniform(swapChances.begin(), swapChances.end(), states, stream);

    combining::crossover(
        beginX,
        endX,
        beginY,
        swapChances.begin(),
        functors::ThresholdLess{0.5},
        beginOut1,
        beginOut2,
        stream);
}

} // namespace random
} // namespace device
