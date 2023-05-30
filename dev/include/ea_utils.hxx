#pragma once
#include "./counter.hxx"
#include "./matrix_utils.hxx"
#include "./rnd_utils.hxx"
#include "./sort_utils.hxx"
#include "./utils.hxx"
#include <algorithm>
#include <random>
#include <vector>

template <typename T, typename PRNG>
[[nodiscard]] inline std::vector<T>
create_rnd_solution_population_tsp(const int nIndividuals, const int nLocations, PRNG& prng) {
    // starting from point 1, not 0 (0 is implicit beginning and ending of a solution)
    const auto     solutionLength = (nLocations - 1);
    std::vector<T> population(solutionLength * nIndividuals);

    const auto end = population.data() + solutionLength * nIndividuals;

    // individual - pointer to the first gene of an individual
    for (auto individual{population.data()}; individual < end; individual += solutionLength) {
        // fill with [1, ..., nGenes - 1)
        // starting from 1, because for closed TSP we implicitly assume 0
        // to be both starting and ending point, thus solution only consists of
        // destination points, excluding 0 at the end
        for (int i{0}; i < nLocations - 1; ++i) {
            individual[i] = static_cast<T>(i + 1);
        }

        std::shuffle(individual, individual + solutionLength, prng);
    }

    return population;
}

// after prng - helper preallocated vectors
template <typename IterIn, typename PRNG>
inline void
mutatePopulation(
    IterIn begin, const std::size_t populationSize, const std::size_t nGenes, const float mutationChance, PRNG& prng, std::vector<char>& mask) {

    for (auto individual{begin}; individual < begin + populationSize * nGenes; individual += nGenes)
        choice_shuffle(mutationChance, individual, individual + nGenes, prng, mask);
}

using CostT  = float;
using CostMx = std::vector<CostT>;

template <typename PRNG>
void
initialize_rnd_symmetric_cost_mx_tsp(CostMx& mx, const int n, PRNG& prng, const CostT minCost = 1e-2, const CostT maxCost = 1e1) {
    std::uniform_real_distribution<CostT> dist{minCost, maxCost};

    for (int i{0}; i < n; ++i) {
        // diagonal
        mx[i * n + i] = std::numeric_limits<CostT>::infinity();

        for (int j{i + 1}; j < n; ++j) {
            const auto cost = dist(prng);

            mx[i * n + j] = cost;
            mx[j * n + i] = cost;
        }
    }
}

template <typename PRNG>
inline CostMx
create_rnd_symmetric_cost_mx_tsp(const int n, PRNG& prng, const CostT minCost = 1e-2, const CostT maxCost = 1e1) {
    CostMx mx(square(n));
    initialize_rnd_symmetric_cost_mx_tsp(mx, n, prng, minCost, maxCost);

    return mx;
}

/// Open TSP loss function - does not add cost from the last destination to the 0 index
template <typename SolutionIter>
[[nodiscard]] inline float
calcLossOpen(const CostMx& costMx, const std::size_t nLocations, SolutionIter begin, SolutionIter end) {
    float loss = 0.0f;

    using Location = std::remove_const<decltype(*begin)>::type;

    // assuming that the starting point is 0
    // Location current = 0;
    int current = 0;

    using CostT    = CostMx::value_type;
    using CostIter = decltype(costMx.begin());
    const MatrixView<CostT, CostIter> mxView(costMx.begin(), nLocations);

    std::for_each(begin, end, [=, &current, &mxView, &loss](Location dst) {
        loss += mxView.get(current, dst);
        current = dst;
    });

    return loss;
}

template <typename SolutionIter>
[[nodiscard]] inline float
calcLossClosed(const CostMx& costMx, const std::size_t nLocations, SolutionIter begin, SolutionIter end) {
    using CostT    = CostMx::value_type;
    using CostIter = decltype(costMx.begin());
    const MatrixView<CostT, CostIter> mxView(costMx.begin(), nLocations);

    const auto last = *(end - 1);

    return calcLossOpen(costMx, nLocations, begin, end) + mxView.get(last, 0);
}

template <typename MigrantPtr>
inline void
swap_migrants(MigrantPtr m1, MigrantPtr m2, const int nGenes) {
    for (int i{0}; i < nGenes; ++i)
        std::swap(m1[i], m2[i]);
}

template <typename GeneT>
using Individual = GeneT*;

template <typename GeneT>
using Population = std::vector<Individual<GeneT>>;

// p<i>, loss<i> are sorted together by loss<i> in decreasing order
template <typename GeneT, typename LossT>
inline void
migrateBetween(
    Population<GeneT>&  p1,
    Population<GeneT>&  p2,
    const int           nGenes,
    std::vector<LossT>& loss1,
    std::vector<LossT>& loss2,
    const int           nMigrants) {

    using Individual = GeneT*;

    // take n best individuals to migrate
    std::vector<Individual> migrants1(nMigrants), migrants2(nMigrants);
    std::copy(p1.cbegin(), p1.cbegin() + nMigrants, migrants1.begin());
    std::copy(p2.cbegin(), p2.cbegin() + nMigrants, migrants2.begin());

    for (int i{0}; i < nMigrants; ++i) {
        swap_migrants(migrants1[i], migrants2[i], nGenes);
        std::swap(loss1[i], loss2[i]);
    }
}

template <typename Gene>
inline auto
solution_to_individuals(std::vector<Gene>& solution, const int nIndividuals, const int nGenes) {
    using GenePtr = decltype(solution.begin())::value_type*;
    std::vector<GenePtr> individuals(nIndividuals);

    for (int i{0}; i < nIndividuals; ++i)
        individuals[i] = static_cast<GenePtr>(&(*solution.begin())) + i * nGenes;

    return individuals;
}

template <typename Individual>
inline void
calcPopulationLosses(
    std::vector<float>&            losses,
    const std::vector<Individual>& population,
    const CostMx&                  costMx,
    const int                      nLocations) {

    const auto nGenes = nLocations - 1;
    std::transform(
        population.cbegin(), population.cend(), losses.begin(),
        [&](auto ind) { return calcLossClosed(costMx, nLocations, ind, ind + nGenes); });
}

template <typename Individual>
inline std::vector<float>
calcPopulationLosses(
    const std::vector<Individual>& population,
    const CostMx&                  costMx,
    const int                      nLocations) {

    std::vector<float> losses(population.size());
    calcPopulationLosses(losses, population, costMx, nLocations);

    return losses;
}

enum class MigrationDirection {
    LEFT  = -1,
    RIGHT = 1,
};

template <typename PRNG>
inline MigrationDirection
getRandomMigrationDirection(PRNG& prng) {
    return static_cast<MigrationDirection>(getRandomSign(prng));
}

/// lossMatrix - each row corresponds to population, each column to individual solution
template <typename GeneT>
inline void
migrate(
    std::vector<Population<GeneT>>&  populations,
    std::vector<std::vector<float>>& lossMatrix,
    const int                        nIndividuals,
    const int                        nGenes,
    const int                        nMigrants,
    const MigrationDirection         migrationDirection) {

    const int nPopulations = populations.size();

    // Treat the first or last population as temporary buffer
    // to avoid unnecessary memory allocation
    // buffer is needed to ensure correct migration scheme,
    // otherwise some individuals may migrate further.
    // The first or last, depending on migration (iteration) direction.
    const auto tempIdx = (nPopulations - 1) * (migrationDirection == MigrationDirection::LEFT);

    auto& tempPopulation     = populations[tempIdx];
    auto& tempPopulationLoss = lossMatrix[tempIdx]; // extracts row
                                                    // pointer to avoid copy

    constexpr bool REORDER_LOSS = true;

    // Sort individuals in each population by their loss function values
    // in decreasing order.
    // Loss vectors also change order.
    // Lazy sorting (while migrating between two populations) introduces bug:
    // temp population (and loss) is unnecessarily sorted each time,
    // but we expect that after sorting, each k migrants appear at the first
    // k indices of the vector.
    for (int i{0}; i < nPopulations; ++i) {
        const auto p    = &populations[i];
        const auto loss = &lossMatrix[i];

        using PopIter  = decltype(p->begin());
        using LossIter = decltype(loss->begin());

        sort_by2<PopIter, LossIter, REORDER_LOSS, SortOrder::DECR>(p->begin(), p->begin() + nIndividuals, loss->begin());
    }

    // for correct iteration direction
    // offset == 1 and nPopulations - 1, because the first population is used as
    // a temporary buffer, so swapping with it is already implicitly
    // handled
    const Counter counter(1, nPopulations, static_cast<int>(migrationDirection));

    for (const auto i : counter) {
        // migrate between next population and temp buffer to preserve
        // migrants that are currently overwritten
        // temp is already "initialized"
        // temp contains previously overwritten migrants, thus swapping with
        // temp "advances" migration by one complete step
        migrateBetween(populations[i], tempPopulation, nGenes, lossMatrix[i], tempPopulationLoss, nMigrants);
    }
}
