#include "./dev-scratchbook.hxx"
#include "ev_alg/crossover.hxx"
#include "ev_alg/ea_utils.hxx"
#include "ev_alg/loss.hxx"
#include "ev_alg/population_generation.hxx"
#include "ev_alg/repair.hxx"
#include "ev_alg/stop_cond.hxx"
#include "iter_utils.hxx"
#include "matrix_utils.hxx"
#include "rnd_utils.hxx"
#include <bits/iterator_concepts.h>
#include <iterator>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>

struct NoOp {};

template <typename T>
concept is_noop_t = std::is_same_v<T, NoOp>;

template <typename T>
[[nodiscard]] inline constexpr bool
is_noop(T const&) noexcept {
    return is_noop_t<T>;
}

struct EvAlgParams {
    const CostMx costMx;
    const int    populationSize;
    const int    nLocations;
    const int    nIslands;
    const int    iterationsPerEpoch;
    const int    nGenes;

    EvAlgParams() = delete;
    [[nodiscard]] EvAlgParams(const CostMx costMx, const int populationSize,
                              const int nLocations, const int nIslands,
                              const int iterationsPerEpoch,
                              const int nGenes) noexcept
    : costMx(costMx),
      populationSize(populationSize),
      nLocations(nLocations),
      nIslands(nIslands),
      iterationsPerEpoch(iterationsPerEpoch),
      nGenes(nGenes) {}
};

struct HistoryEntry {
    const float minLoss;
    const float maxLoss;
    const float meanLoss;
    const float stddev;
};

template <typename RndPopulationGenerator, typename MutationOp,
          typename CrossOp, typename LossFn, typename MigrationOp,
          typename StopCondition, typename PRNG, typename RepairOp = NoOp,
          typename Coder = NoOp>
struct IslandEvolutionaryAlgorithm {
    using GeneT                 = int;
    using PopulationLossHistory = std::vector<HistoryEntry>;
    using PopulationMemory      = std::vector<GeneT>;

    IslandEvolutionaryAlgorithm() = delete;
    [[nodiscard]] IslandEvolutionaryAlgorithm(
        RndPopulationGenerator rndPopulationGenerator, MutationOp mutationOp,
        CrossOp crossOp, LossFn lossFn, MigrationOp migrationOp,
        StopCondition stopCondition, PRNG prng, EvAlgParams params,
        RepairOp repairOp = NoOp(), Coder coder = NoOp())
    : rndPopulationGenerator(rndPopulationGenerator),
      mutationOp(mutationOp),
      crossoverOp(crossOp),
      migrationOp(migrationOp),
      stopCondition(stopCondition),
      prng(prng),
      params(params),
      repairOp(repairOp),
      coder(coder),
      islands(params.nIslands) {
        // Copy loss functions, because they have vectors with loss function
        // values. They will be needed for migration for each island.
        lossFunPerIsland.reserve(params.nIslands);
        lossFunPerIsland.push_back(std::move(lossFn));

        for (const auto _ : std::ranges::iota_view(0, params.nIslands - 1))
            lossFunPerIsland.push_back(lossFunPerIsland[0]);
    }

    void
    operator()() {
        initialize_islands();
        run_island(0);
    }

private:
    RndPopulationGenerator rndPopulationGenerator;
    MutationOp             mutationOp;
    CrossOp                crossoverOp;
    std::vector<LossFn>    lossFunPerIsland;
    MigrationOp            migrationOp;
    StopCondition          stopCondition;
    PRNG                   prng;
    const EvAlgParams      params;
    RepairOp               repairOp;
    Coder                  coder;

    std::vector<PopulationMemory> islands;

    // results
    std::vector<PopulationLossHistory> lossHistories;
    std::vector<GeneT>                 bestSolution;

    using PopulationMxView = MatrixView<GeneT*>;

    void
    initialize_islands() {
        for (auto& islandPopulation : islands)
            islandPopulation = std::move(rndPopulationGenerator(prng));
    }

    void
    run_island(const std::size_t islandIdx) {
        auto&            population = islands[islandIdx];
        PopulationMxView populationView(population.data(),
                                        params.populationSize, params.nGenes);

        auto individualPtrsView = solution_to_individuals(
            population, params.populationSize, params.nGenes);

        mutatePopulation(populationView);
        repairPopulation(populationView);
        gradePopulation(islandIdx, individualPtrsView);
        // elitist selection - best solutions at the beginning
        // sorts only pointers, they will be used for crossover
        sortPopulation(islandIdx, individualPtrsView);
        std::cout << '\n';
        std::cout << "before: \n";
        prettyPrintMx(populationView);
        breedPopulation(individualPtrsView);
        std::cout << "after: \n";
        prettyPrintMx(populationView);
    }

    void
    mutatePopulation(PopulationMxView& population) {
        for (auto row : population.rows())
            mutationOp(row.data(), prng);
    }

    void
    repairPopulation(PopulationMxView& population) {
        if constexpr (is_noop_t<RepairOp>)
            return;
        else
            for (auto row : population.rows())
                repairOp(row.begin(), row.end());
    }

    void
    gradePopulation(const std::size_t islandIdx, auto& individualPtrs) {
        lossFunPerIsland[islandIdx](individualPtrs);
    }

    /// individualPtrs - vector of pointers to the beginnings of each individual
    /// solution
    /// This function sorts only pointers, so it is cheap
    void
    sortPopulation(const std::size_t islandIdx, auto& individualPtrs) {
        constexpr bool REORDER_LOSS = true;

        auto& losses = lossFunPerIsland[islandIdx].values;

        using PopIter  = decltype(individualPtrs.begin());
        using LossIter = decltype(losses.begin());

        sort_by2<PopIter, LossIter, REORDER_LOSS, SortOrder::DECR>(
            individualPtrs.begin(), individualPtrs.end(), losses.begin());
    }

    void
    breedPopulation(auto& sortedParentPtrs) {
        for (int i{0}; i < sortedParentPtrs.size(); i += 2) {
            const auto p1Ptr(sortedParentPtrs[i]);
            const auto p2Ptr(sortedParentPtrs[i + 1]);

            crossoverOp(p1Ptr, p1Ptr + params.nGenes, p2Ptr, prng);
        }
    }
};

inline void
show_ev_alg_tsp() {
    const std::size_t prng_seed                 = 0;
    const int         iterationsPerEpoch        = 10;
    const int         nEpochs                   = 6;
    const int         nEpochsWithourImprovement = 2;

    const int   nLocations       = 16;
    const float minCost          = 1e-1f;
    const float maxCost          = 1e1;
    const int   islandPopulation = 16;
    const int   nIslands         = 4;
    // in normal, same-size coding, the first and last (0) location is implicit
    // if special solution coder is used, this will differ
    const int   nGenes         = nLocations - 1;
    const float mutationChance = 0.1f;
    const float migrationRatio = 0.1f;
    const auto  nMigrants =
        static_cast<int>(std::max(1.0f, migrationRatio * islandPopulation));

    auto       prng = create_rng(prng_seed);
    const auto costMx =
        create_rnd_symmetric_cost_mx_tsp(nLocations, prng, minCost, maxCost);

    using LocationIdxT = int;
    // this might differ for special solution coder
    using GeneT = LocationIdxT;

    static_assert(std::same_as<GeneT, int>, "non-int genes not supported");

    RndPopulationGeneratorOTSP populationGenerator{islandPopulation,
                                                   nLocations};
    Mutator                    mutator(nGenes, mutationChance);
    CrossoverPMX2PointAdapter  crossover{};
    LossFnClosed               lossFn(nLocations, &costMx, islandPopulation);
    MigrationOp<GeneT>         migrator(islandPopulation, nGenes, nMigrants);
    StopCondition     stopCondition(nEpochs, nEpochsWithourImprovement);
    const EvAlgParams params(costMx, islandPopulation, nLocations, nIslands,
                             iterationsPerEpoch, nGenes);
    // those are optional
    auto repairOp = NoOp();
    auto coder    = NoOp();

    // PMX
    IslandEvolutionaryAlgorithm algo(populationGenerator, mutator, crossover,
                                     lossFn, migrator, stopCondition, prng,
                                     params, repairOp, coder);
    algo();
}

int
main() {
    try {
        // playground();
        // showcase_permutation_inversion_sequence();
        // pmx_crossover_showcase();
        show_ev_alg_tsp();
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        throw;
    }
    return 0;
}
