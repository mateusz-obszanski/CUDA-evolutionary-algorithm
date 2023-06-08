#pragma once
#include "./ea_utils.hxx"
#include "./parameters.hxx"
#include "./stats.hxx"
#include <concepts>
#include <fstream>

struct NoOp {};

template <typename T>
concept is_noop_t = std::is_same_v<T, NoOp>;

template <typename T>
[[nodiscard]] inline constexpr bool
is_noop(T const&) noexcept {
    return is_noop_t<T>;
}

using HistoryEntry = StandardStats;

struct PopulationSizeError : public std::exception {
    const char*
    what() const noexcept override {
        return "population size must be an even number";
    }
};

template <typename RndPopulationGenerator, typename MutationOp,
          typename CrossOp, typename LossFn, typename MigrationOp,
          typename StopCondition, typename RepairOp = NoOp,
          typename Coder = NoOp>
struct IslandEvolutionaryAlgorithm {
    using GeneT                 = int;
    using PopulationLossHistory = std::vector<HistoryEntry>;
    using PopulationMemory      = std::vector<GeneT>;
    using LossMx                = std::vector<std::vector<float>>;

    // public, because stop reason might be needed after algorithm completed
    StopCondition stopCondition;

    IslandEvolutionaryAlgorithm() = delete;

    [[nodiscard]] IslandEvolutionaryAlgorithm(
        RndPopulationGenerator rndPopulationGenerator, MutationOp mutationOp,
        CrossOp crossOp, LossFn lossFn, MigrationOp migrationOp,
        StopCondition stopCondition, EvAlgParams params,
        RepairOp repairOp = NoOp(), Coder coder = NoOp())
    : stopCondition(stopCondition),
      rndPopulationGenerator(rndPopulationGenerator),
      mutationOp(mutationOp),
      crossoverOp(crossOp),
      migrationOp(migrationOp),
      params(params),
      repairOp(repairOp),
      coder(coder),
      islands(params.nIslands),
      lossHistories(params.nIslands),
      bestSolution(params.nGenes),
      bestFitness(std::numeric_limits<float>::infinity()),
      lossMx(params.nIslands) {
        if (not isEven(params.populationSize))
            throw PopulationSizeError();

        // Copy loss functions, because they have vectors with loss function
        // values. They will be needed for migration for each island.
        lossFunPerIsland.reserve(params.nIslands);
        lossFunPerIsland.push_back(std::move(lossFn));

        for ([[maybe_unused]] const auto _ :
             std::ranges::iota_view(0u, params.nIslands - 1)) {
            lossFunPerIsland.push_back(lossFunPerIsland[0]);
        }

        for (auto& islandLosses : lossMx) {
            islandLosses.resize(params.populationSize);
        }
    }

    template <typename PRNG>
    void
    operator()(CostMx const& costMx, PRNG& prng) {
        initialize_islands(prng);

        const std::ranges::iota_view islandIdxs(0u, params.nIslands);
        const std::ranges::iota_view epochIters(0u, params.iterationsPerEpoch);

        auto individualsMx = create_individual_view_matrix();

        // check stop condition between migrations
        do {
            // this loop could be parallel, if saving the best solution and
            // its fitness are thread-safe operations
            for (const auto islandIdx : islandIdxs) {
                run_island(costMx, prng, islandIdx, individualsMx[islandIdx]);
                // at the end of run_island, population is bred, so
                // repairing and loss recalculation are necessary

                if constexpr (not is_noop_t<RepairOp>) {
                    PopulationMxView populationView(islands[islandIdx].data(),
                                                    params.populationSize,
                                                    params.nGenes);
                    repair_population(populationView);
                }

                grade_population(costMx, islandIdx, individualsMx[islandIdx]);
                sort_population(islandIdx, individualsMx[islandIdx]);

                remember_best_solution(islandIdx, individualsMx[islandIdx]);
                remember_stats(islandIdx);
            }

            migrationOp(individualsMx, lossMx, prng);

        } while (!stopCondition(bestFitness));
    }

    void
    save_results(std::filesystem::path dirpath) const {
        std::filesystem::create_directories(dirpath);

        for (std::size_t i{0}; i < lossHistories.size(); ++i) {
            const auto filename =
                dirpath / ("island_" + std::to_string(i) + ".dat");
            std::ofstream file(filename, std::ios::binary);

            auto const& islandHistory = lossHistories[i];

            for (auto const& entry : islandHistory)
                entry.write_binary(file);
        }
    }

    [[nodiscard]] auto
    get_best_fitness() const noexcept {
        return bestFitness;
    }

    [[nodiscard]] auto
    get_best_solution() const {
        return bestSolution;
    }

private:
    RndPopulationGenerator rndPopulationGenerator;
    MutationOp             mutationOp;
    CrossOp                crossoverOp;
    std::vector<LossFn>    lossFunPerIsland;
    MigrationOp            migrationOp;
    const EvAlgParams      params;
    RepairOp               repairOp;
    Coder                  coder;

    std::vector<PopulationMemory> islands;

    // results
    std::vector<PopulationLossHistory> lossHistories;
    std::vector<GeneT>                 bestSolution;
    float                              bestFitness;

    LossMx lossMx;

    using PopulationMxView = MatrixView<GeneT*>;

    template <typename PRNG>
    void
    initialize_islands(PRNG& prng) {
        for (auto& islandPopulation : islands) {
            const auto population = rndPopulationGenerator(prng);

            if constexpr (is_noop_t<Coder>)
                islandPopulation = std::move(population);
            else
                islandPopulation = std::move(coder.encode_many(population));
        }
    }

    // matrix [islandIdx x populationSize] of pointers to the beginnings of each
    // individual solution
    [[nodiscard]] std::vector<IndividualPtrsView<GeneT>>
    create_individual_view_matrix() {
        std::vector<IndividualPtrsView<GeneT>> individualsMx(params.nIslands);

        const std::ranges::iota_view<std::size_t, std::size_t> islandIdxs(
            0, params.nIslands);

        // initialize pointers to individual solutions for each island
        for (const auto islandIdx : islandIdxs)
            individualsMx[islandIdx] = solution_to_individuals(
                islands[islandIdx], params.populationSize, params.nGenes);

        return individualsMx;
    }

    template <typename PRNG>
    void
    run_island(CostMx const& costMx, PRNG& prng, const unsigned int islandIdx,
               IndividualPtrsView<GeneT>& individualPtrsView) {

        auto&            population = islands[islandIdx];
        PopulationMxView populationView(population.data(),
                                        params.populationSize, params.nGenes);

        for (unsigned int i{0}; i < params.iterationsPerEpoch; ++i) {
            mutate_population(populationView, prng);
            repair_population(populationView);
            grade_population(costMx, islandIdx, individualPtrsView);
            // elitist selection - best solutions at the beginning
            // sorts only pointers, they will be used for crossover
            sort_population(islandIdx, individualPtrsView);
            remember_best_solution(islandIdx, individualPtrsView);
            remember_stats(islandIdx);
            breed_population(individualPtrsView, prng);
        }
    }

    template <typename PRNG>
    void
    mutate_population(PopulationMxView& population, PRNG& prng) {
        for (auto row : population.rows())
            mutationOp(row.data(), prng);
    }

    /// for more advanced (but slow) repair procedure, cost matrix might be
    /// required
    void
    repair_population(PopulationMxView& population) {
        if constexpr (is_noop_t<RepairOp>)
            return;
        else
            for (auto row : population.rows())
                repairOp(row.begin(), row.end());
    }

    void
    grade_population(CostMx const& costMx, const unsigned int islandIdx,
                     IndividualPtrsView<GeneT>& individualPtrs) {
        if constexpr (is_noop_t<Coder>)
            lossFunPerIsland[islandIdx](individualPtrs, costMx,
                                        lossMx[islandIdx]);
        else {
            // if solutions are encoded, loss function must get decoded version
            auto decoded = coder.decode_many(islands[islandIdx]);
            // ignore individualPtrs, recalculate them for decoded data
            auto decodedIndividualPtrs = solution_to_individuals(
                decoded, params.populationSize, coder.decoded_length());

            lossFunPerIsland[islandIdx](decodedIndividualPtrs, costMx,
                                        lossMx[islandIdx]);
        }
    }

    void
    remember_best_solution(
        const std::size_t                islandIdx,
        IndividualPtrsView<GeneT> const& individualPtrsView) {
        // assumes that individualPtrsView has already been sorted together with
        // loss function values

        if (const auto currentBestLoss = lossMx[islandIdx][0];
            currentBestLoss < bestFitness) {

            bestFitness = currentBestLoss;

            const auto currentBestSolutionBegin = individualPtrsView[0];

            std::copy(currentBestSolutionBegin,
                      currentBestSolutionBegin + params.nGenes,
                      bestSolution.begin());
        }
    }

    /// individualPtrs - vector of pointers to the beginnings of each individual
    /// solution
    /// This function sorts only pointers, so it is cheap
    void
    sort_population(const std::size_t          islandIdx,
                    IndividualPtrsView<GeneT>& individualPtrs) {
        constexpr bool REORDER_LOSS = true;

        auto& losses = lossMx[islandIdx];

        using PopIter  = decltype(individualPtrs.begin());
        using LossIter = decltype(losses.begin());

        sort_by2<PopIter, LossIter, REORDER_LOSS, SortOrder::DECR>(
            individualPtrs.begin(), individualPtrs.end(), losses.begin());
    }

    void
    remember_stats(const std::size_t islandIdx) {
        auto& losses = lossMx[islandIdx];

        lossHistories[islandIdx].push_back(
            HistoryEntry::from_iter(losses.cbegin(), losses.cend()));
    }

    /// Assumes that population size is even, if not, happy SEGFAULT day
    template <typename PRNG>
    void
    breed_population(IndividualPtrsView<GeneT>& sortedParentPtrs, PRNG& prng) {
        for (std::size_t i{0}; i < sortedParentPtrs.size(); i += 2) {
            const auto p1Ptr(sortedParentPtrs[i]);
            const auto p2Ptr(sortedParentPtrs[i + 1]);

            crossoverOp(p1Ptr, p1Ptr + params.nGenes, p2Ptr, prng);
        }
    }
};
