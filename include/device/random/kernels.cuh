#pragma once

#include "../../types/concepts.hxx"
#include "../../types/types.hxx"
#include <curand_kernel.h>

namespace device {
namespace random {
namespace kernel {

using seed_t     = ullong;
using sequence_t = ullong;
using offset_t   = ullong;

template <typename State>
concept IsInitializableRndState =
    ::types::concepts::AnyOf<
        State, curandStateXORWOW_t, curandStateMRG32k3a_t, curandStatePhilox4_32_10_t>;

template <IsInitializableRndState State>
__global__ void
setup_kernel(State* state, seed_t seed = 0, sequence_t first_seq = 0, offset_t offset = 0) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number
    curand_init(1234, first_seq + id, offset, &state[id]);
}

} // namespace kernel
} // namespace random
} // namespace device
