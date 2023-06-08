#pragma once
#include <algorithm>
#include <vector>

template <typename PRNG, typename NodeIdx = int>
[[nodiscard]] inline std::vector<NodeIdx>
create_rnd_solution_population_tsp(const int nIndividuals, const int nLocations,
                                   PRNG& prng) {
    // starting from point 1, not 0 (0 is implicit beginning and ending of a
    // solution)
    const auto           solutionLength = (nLocations - 1);
    std::vector<NodeIdx> population(solutionLength * nIndividuals);

    const auto end = population.data() + solutionLength * nIndividuals;

    // individual - pointer to the first gene of an individual
    for (auto individual{population.data()}; individual < end;
         individual += solutionLength) {
        // fill with [1, ..., nGenes - 1)
        // starting from 1, because for closed TSP we implicitly assume 0
        // to be both starting and ending point, thus solution only consists of
        // destination points, excluding 0 at the end
        for (int i{0}; i < nLocations - 1; ++i) {
            individual[i] = static_cast<NodeIdx>(i + 1);
        }

        std::shuffle(individual, individual + solutionLength, prng);
    }

    return population;
}

struct RndPopulationGeneratorOTSP {
    const unsigned int nIndividuals;
    const unsigned int nLocations;

    template <typename PRNG, typename NodeIdx = int>
    std::vector<NodeIdx>
    operator()(PRNG& prng) const {
        return create_rnd_solution_population_tsp<PRNG, NodeIdx>(
            nIndividuals, nLocations, prng);
    }
};
