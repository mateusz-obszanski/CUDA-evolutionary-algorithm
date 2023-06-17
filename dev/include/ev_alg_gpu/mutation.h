#pragma once

#include "../gpu/random.h"
#include "../gpu/random.h"

struct MutatorGPU {
    const unsigned int nGenes;
    const float        mutationChance;

    [[nodiscard]] MutatorGPU(unsigned int nGenes, float mutationChance)
    : nGenes(nGenes), mutationChance(mutationChance) {}

    [[nodiscard]] MutatorGPU(MutatorGPU const& other)
    : MutatorGPU(other.nGenes, other.mutationChance) {}

    [[nodiscard]] MutatorGPU(MutatorGPU&& other)
    : nGenes(other.nGenes), mutationChance(other.mutationChance) {}

    template <typename Iter, typename ThrustRndEngine,
              device::random::IsInitializableRndState State = curandState,

              template <typename TT>
              typename Allocator = device::memory::allocator::DeviceAllocator>
    void
    operator()(Iter sequence, ThrustRndEngine& prng,
               device::random::RndStateMemory<State, Allocator>& rndStates) {
        device::random::shuffle_with_prob(sequence, sequence + nGenes,
                                          mutationChance, rndStates, prng);
    }
};
