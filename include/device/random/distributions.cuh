#pragma once

#include "../errors.cuh"
#include "../memory/allocator.cuh"
#include "./kernels.cuh"
#include <concepts>
#include <cuda/std/bit>
#include <curand_kernel.h>
#include <type_traits>
#include <vector>

namespace device {
namespace random {

using kernel::offset_t;
using kernel::seed_t;
using kernel::sequence_t;

namespace {

using kernel::IsInitializableRndState;

}

template <
    IsInitializableRndState State = curandState,

    template <typename TT>
    typename Alloc = device::memory::allocator::DeviceAllocator>

    requires device::memory::allocator::concepts::DeviceAllocator<Alloc<State>>
using RndStateMemory = device::memory::raii::SimpleMemory<State, Alloc>;

template <std::floating_point T, IsInitializableRndState State>
__device__ inline T
generate_uniform(State* state) noexcept {
    if constexpr (std::is_same_v<T, double>)
        return curand_uniform_double(state);
    else
        return curand_uniform(state);
}

// TODO DELETE BaseDistribution, just write templated functions that take random state, You own the state data in a separate container
// TODO thrust function for initializing allocated random states (for_each)
// TODO DistributionPool class
// TODO (optional) NormalDistributionPoolHeterogenous - with differend m and sigma for each distribution

template <typename Distribution>
struct DistributionTraits;

template <typename DistributionSubclass>
class BaseDistribution {
public:
    using result_type       = typename DistributionTraits<DistributionSubclass>::result_type;
    using state_type        = typename DistributionTraits<DistributionSubclass>::state_type;
    using distribution_type = DistributionSubclass;

    state_type* state;

    [[nodiscard]] __host__ __device__
    BaseDistribution(seed_t seed = 0, sequence_t seq = 0, offset_t offset = 0) noexcept {
#ifdef __CUDA_ARCH__
        init_device(seed, seq, offset);
#else
        init_host(seed, seq, offset);
#endif
    };

    [[nodiscard]] __host__ __device__ explicit BaseDistribution(
        seed_t seed, sequence_t seq, offset_t offset, cudaStream_t stream) noexcept {}

    [[nodiscard]] __host__ __device__
    BaseDistribution(const DistributionSubclass& other) noexcept : state{other.state} {}

    [[nodiscard]] __host__ __device__
    BaseDistribution(DistributionSubclass&& other) noexcept : state{std::move(other.state)} {}

    /// Assumes that rndState has already been initialized on device
    [[nodiscard]] __host__ __device__ explicit BaseDistribution(
        state_type* rndState) noexcept : state{state} {}

    __host__ void
    init_host(seed_t seed = 0, sequence_t seq = 0, offset_t offset = 0, cudaStream_t stream = 0) {
        kernel::setup_kernel<<<1, 1, 0, stream>>>(state, seed, seq, offset);
        errors::check();
    }

    __device__ void
    init_device(seed_t seed = 0, sequence_t seq = 0, offset_t offset = 0) {
        curand_init(seed, seq, offset, state);
    }

    __device__ void
    init_device(seed_t seed, sequence_t seq, offset_t offset, cudaStream_t stream) {
        kernel::setup_kernel<<<1, 1, 0, stream>>>(state, seed, seq, offset);
        printf("WARNING are you sure?\n");
    }

    [[nodiscard]] __device__ result_type
    generate() noexcept {
        return static_cast<DistributionSubclass*>(this)->generate();
    }
};

template <std::floating_point T = float, IsInitializableRndState State = curandState>
class UniformDistribution : public BaseDistribution<UniformDistribution<T, State>> {
private:
    friend class BaseDistribution<UniformDistribution>;

    __device__ T
    _generate() noexcept {
        return generate_uniform<T, State>(this->state);
    }
};

template <std::floating_point T, IsInitializableRndState State>
struct DistributionTraits<UniformDistribution<T, State>> {
    using state_type  = State;
    using result_type = T;
};

namespace concepts {

template <typename D>
concept IsDistribution = std::derived_from<D, BaseDistribution<D>>;

}

namespace {

using concepts::IsDistribution;

}

template <
    IsDistribution          Distribution,
    IsInitializableRndState State = curandState,
    typename HostAlloc            = std::pmr::polymorphic_allocator<State>>
class DistributionPool {
private:
    cuda::std::vector<Distribution>
};

} // namespace random
} // namespace device
