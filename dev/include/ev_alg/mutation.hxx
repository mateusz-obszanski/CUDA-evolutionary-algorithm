#pragma once
#include "../algo_utils.hxx"
#include <utility>
#include <vector>

class Mutator {
public:
    const unsigned int nGenes;
    const float        mutationChance;

    [[nodiscard]] Mutator(unsigned int nGenes, const float mutationChance)
    : nGenes(nGenes),
      mutationChance(mutationChance),
      preallocated_mask(nGenes) {}

    [[nodiscard]] Mutator(Mutator const& other)
    : Mutator(other.nGenes, other.mutationChance) {}
    [[nodiscard]] Mutator(Mutator&& other)
    : nGenes(other.nGenes),
      mutationChance(other.mutationChance),
      preallocated_mask(std::move(other.preallocated_mask)) {}

    template <typename Iter, typename PRNG>
    void
    operator()(Iter sequence, PRNG& prng) {
        choice_shuffle(sequence, sequence + nGenes, mutationChance, prng,
                       preallocated_mask);
    }

private:
    std::vector<char> preallocated_mask;
};

// after prng - helper preallocated vectors
template <typename IterIn, typename PRNG>
inline void
mutatePopulation(IterIn begin, const unsigned int populationSize,
                 const unsigned int nGenes, const float mutationChance,
                 PRNG& prng, std::vector<char>& mask) {

    for (auto individual{begin}; individual < begin + populationSize * nGenes;
         individual += nGenes)
        choice_shuffle(individual, individual + nGenes, mutationChance, prng,
                       mask);
}
