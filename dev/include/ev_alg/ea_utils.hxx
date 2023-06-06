#pragma once
#include "../algo_utils.hxx"
#include "../iter_utils.hxx"
#include "../matrix_utils.hxx"
#include "../rnd_utils.hxx"
#include "../sort_utils.hxx"
#include "../utils.hxx"
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>

template <typename Gene>
using IndividualPtrsView = std::vector<Gene*>;

template <typename Gene>
inline IndividualPtrsView<Gene>
solution_to_individuals(std::vector<Gene>& solution,
                        const unsigned int nIndividuals,
                        const unsigned int nGenes) {
    using GenePtr = decltype(solution.begin())::value_type*;
    std::vector<GenePtr> individuals(nIndividuals);

    for (unsigned int i{0}; i < nIndividuals; ++i)
        individuals[i] =
            static_cast<GenePtr>(&(*solution.begin())) + i * nGenes;

    return individuals;
}

using CostT  = float;
using CostMx = std::vector<CostT>;

template <typename PRNG>
void
initialize_rnd_symmetric_cost_mx_tsp(CostMx& mx, const unsigned int n,
                                     PRNG& prng, const CostT minCost = 1e-2f,
                                     const CostT maxCost = 1e1f) {
    std::uniform_real_distribution<CostT> dist{minCost, maxCost};

    for (unsigned int i{0}; i < n; ++i) {
        // diagonal
        mx[i * n + i] = std::numeric_limits<CostT>::infinity();

        for (unsigned int j{i + 1}; j < n; ++j) {
            const auto cost = dist(prng);

            mx[i * n + j] = cost;
            mx[j * n + i] = cost;
        }
    }
}

template <typename PRNG>
inline CostMx
create_rnd_symmetric_cost_mx_tsp(const unsigned int n, PRNG& prng,
                                 const CostT minCost = 1e-2f,
                                 const CostT maxCost = 1e1f) {
    CostMx mx(square(n));
    initialize_rnd_symmetric_cost_mx_tsp(mx, n, prng, minCost, maxCost);

    return mx;
}
