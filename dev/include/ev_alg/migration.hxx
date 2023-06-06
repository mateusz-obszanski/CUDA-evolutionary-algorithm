#pragma once
#include "../sort_utils.hxx"
#include <iostream>
#include <utility>
#include <vector>

template <typename MigrantPtr>
inline void
swap_migrants(MigrantPtr m1, MigrantPtr m2, const int nGenes) {
    for (int i{0}; i < nGenes; ++i)
        std::swap(m1[i], m2[i]);
}

template <typename GeneT>
using IndividualPtr = GeneT*;

template <typename GeneT>
using Population = std::vector<IndividualPtr<GeneT>>;

// p<i>, loss<i> are sorted together by loss<i> in decreasing order
template <typename GeneT, typename LossT>
inline void
migrate_between(Population<GeneT>& p1, Population<GeneT>& p2, const int nGenes,
                std::vector<LossT>& loss1, std::vector<LossT>& loss2,
                const int nMigrants) {

    // take n best individuals to migrate
    // std::vector<IndividualPtr<GeneT>> migrants1(nMigrants),
    // migrants2(nMigrants);
    // std::copy(p1.cbegin(), p1.cbegin() + nMigrants, migrants1.begin());
    // std::copy(p2.cbegin(), p2.cbegin() + nMigrants, migrants2.begin());

    // take n best individuals to migrate
    for (int i{0}; i < nMigrants; ++i) {
        // swap_migrants(migrants1[i], migrants2[i], nGenes);
        swap_migrants(p1[i], p2[i], nGenes);
        std::swap(loss1[i], loss2[i]);
        std::swap(p1[i], p2[i]);
    }
}

enum class MigrationDirection {
    LEFT  = -1,
    RIGHT = 1,
};

inline constexpr std::string
stringify_migration_direction_short(MigrationDirection d) noexcept {
    return (d == MigrationDirection::LEFT) ? "LEFT" : "RIGHT";
}

template <typename PRNG>
inline MigrationDirection
get_random_migration_direction(PRNG& prng) {
    return static_cast<MigrationDirection>(getRandomSign(prng));
}

/// lossMatrix - each row corresponds to population, each column to individual
/// solution
template <typename GeneT>
inline void
migrate(std::vector<Population<GeneT>>&  populations,
        std::vector<std::vector<float>>& lossMatrix,
        const unsigned int nIndividuals, const unsigned int nGenes,
        const unsigned int       nMigrants,
        const MigrationDirection migrationDirection) {

    const auto nPopulations = populations.size();

    // Treat the first or last population as temporary buffer
    // to avoid unnecessary memory allocation
    // buffer is needed to ensure correct migration scheme,
    // otherwise some individuals may migrate further.
    // The first or last, depending on migration (iteration) direction.
    const auto isMigrationToLeft =
        (migrationDirection == MigrationDirection::LEFT);
    const auto tempIdx = (nPopulations - 1) * isMigrationToLeft;

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
    for (std::size_t i{0}; i < nPopulations; ++i) {
        const auto p    = &populations[i];
        const auto loss = &lossMatrix[i];

        using PopIter  = decltype(p->begin());
        using LossIter = decltype(loss->begin());

        sort_by2<PopIter, LossIter, REORDER_LOSS, SortOrder::DECR>(
            p->begin(), p->begin() + nIndividuals, loss->begin());
    }

    const bool isMigrationToRight = not isMigrationToLeft;

    // count up or down (from 1 upwards or from nPopulations - 2 downwards),
    // depending on migration direction
    // offset of 1, because the first population serves as swap buffer
    const int step  = static_cast<int>(migrationDirection);
    const int start = static_cast<int>(tempIdx) + step;
    const int stop =
        -1 + (static_cast<int>(nPopulations) + 1) * isMigrationToRight;

    // this loop MUST be sequential or other migration method (with
    // temporary buffer) must be used
    for (int i{start}; i != stop; i += step) {
        // migrate between next population and temp buffer to preserve
        // migrants that are currently overwritten
        // temp is already "initialized"
        // temp contains previously overwritten migrants, thus swapping with
        // temp "advances" migration by one complete step

        migrate_between(populations[i], tempPopulation, nGenes, lossMatrix[i],
                        tempPopulationLoss, nMigrants);
    }
}

template <typename GeneT>
class MigrationOp {
public:
    const unsigned int nIndividuals;
    const unsigned int nGenes;
    const unsigned int nMigrants;

    MigrationOp() = delete;
    [[nodiscard]] MigrationOp(unsigned int nIndividuals, unsigned int nGenes,
                              unsigned int nMigrants) noexcept
    : nIndividuals(nIndividuals), nGenes(nGenes), nMigrants(nMigrants) {}
    [[nodiscard]] MigrationOp(MigrationOp const&) = default;
    [[nodiscard]] MigrationOp(MigrationOp&&)      = default;

    template <typename PRNG>
    void
    operator()(std::vector<Population<GeneT>>&  populations,
               std::vector<std::vector<float>>& lossMatrix, PRNG& prng) const {

        const auto direction = get_random_migration_direction(prng);
        migrate(populations, lossMatrix, nIndividuals, nGenes, nMigrants,
                direction);
    }
};
