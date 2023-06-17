#pragma once
#include "../ev_alg/ea_utils.hxx"
#include "../ev_alg/ev_alg.hxx"
#include "../ev_alg/parameters.hxx"
#include "../ev_alg/stats.hxx"
#include "../gpu/errors.hxx"
#include <concepts>
#include <fstream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

template <typename RndPopulationGenerator, typename MutationOp,
          typename CrossOp, typename LossFn, typename MigrationOp,
          typename StopCondition, typename RepairOp = NoOp,
          typename Coder = NoOp>
struct IslandEvolutionaryAlgorithmGPU {
    using GeneT                 = int;
    using PopulationLossHistory = std::vector<HistoryEntry>;
    using PopulationMemory      = std::vector<GeneT>;
    using LossMx                = std::vector<std::vector<float>>;

    // public, because stop reason might be needed after algorithm completed
    StopCondition stopCondition;

    IslandEvolutionaryAlgorithmGPU() = delete;

    [[nodiscard]] IslandEvolutionaryAlgorithmGPU(
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

    template <typename PRNG, typename THRUST_PRNG, typename CURAND_RND_STATES>
    void
    operator()(CostMx const& costMx, PRNG& prng, THRUST_PRNG& thrustPrng,
               CURAND_RND_STATES& states) {
        initialize_islands(prng);

        auto individualsMx = create_individual_view_matrix();

        unsigned epoch = 0;

        // check stop condition between migrations
        do {
            std::cout << "epoch " << epoch++ << '\n';
            run_epoch(costMx, prng, thrustPrng, states, individualsMx);

        } while (!stopCondition(bestFitness));
    }

    template <typename PRNG, typename THRUST_PRNG, typename CURAND_RND_STATES>
    void
    run_epoch(CostMx const& costMx, PRNG& prng, THRUST_PRNG& thrustPrng,
              CURAND_RND_STATES&                      states,
              std::vector<IndividualPtrsView<GeneT>>& individualsMx) {
        const std::ranges::iota_view islandIdxs(0u, params.nIslands);

        // initialization of device memory

        // this loop could be parallel, if saving the best solution and
        // its fitness are thread-safe operations
        for (const auto islandIdx : islandIdxs) {
            // gpu, results on host
            run_island(costMx, prng, thrustPrng, states, islandIdx,
                       individualsMx[islandIdx]);
            // at the end of run_island, population is bred, so
            // repairing and loss recalculation are necessary

            // this does not step into if
            if constexpr (not is_noop_t<RepairOp>) {
                PopulationMxView populationView(islands[islandIdx].data(),
                                                params.populationSize,
                                                params.nGenes);
                repair_population(populationView);
            }

            // gpu
            grade_population(costMx, islandIdx, individualsMx[islandIdx]);
            // gpu
            sort_population(islandIdx, individualsMx[islandIdx]);

            // gpu -> host
            remember_best_solution(islandIdx, individualsMx[islandIdx]);
            // calc on gpu -> host
            remember_stats(islandIdx);
        }

        // TODO: data gpu -> host

        // host
        migrationOp(individualsMx, lossMx, prng);
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

    template <typename PRNG, typename THRUST_PRNG, typename CURAND_RND_STATES>
    void
    run_island(CostMx const& costMx, PRNG& prng, THRUST_PRNG& thrustPrng,
               CURAND_RND_STATES& states, const unsigned int islandIdx,
               IndividualPtrsView<GeneT>& individualPtrsView) {

        auto&            population = islands[islandIdx];
        PopulationMxView populationView(population.data(),
                                        params.populationSize, params.nGenes);

        for (unsigned int i{0}; i < params.iterationsPerEpoch; ++i) {
            mutate_population(populationView, thrustPrng, states);
            repair_population(populationView);
            grade_population(costMx, islandIdx, individualPtrsView);
            // elitist selection - best solutions at the beginning
            // sorts only pointers, they will be used for crossover
            sort_population(islandIdx, individualPtrsView);
            remember_best_solution(islandIdx, individualPtrsView);
            remember_stats(islandIdx);
            breed_population(individualPtrsView, states);
        }
    }

    template <typename THRUST_PRNG, typename CURAND_RND_STATES>
    void
    mutate_population(PopulationMxView& population, THRUST_PRNG& prng,
                      CURAND_RND_STATES& states) {
        // to gpu
        thrust::device_vector<GeneT> gpuPopulation(population.size());
        thrust::copy(population.data(), population.data() + population.size(),
                     gpuPopulation.begin());

        for (int i{0}; i < population.height(); ++i) {
            mutationOp(gpuPopulation.data() + i * population.width(), prng,
                       states);
        }

        // to host
        thrust::copy(gpuPopulation.cbegin(), gpuPopulation.cend(),
                     population.data());
    }

    /// for more advanced (but slow) repair procedure, cost matrix might be
    /// required
    void
    repair_population(PopulationMxView& population) {
        if constexpr (is_noop_t<RepairOp>)
            return;
        else {
            errors::throwNotImplemented(__LINE__);
            for (auto row : population.rows())
                repairOp(row.begin(), row.end());
        }
    }

    void
    grade_population(CostMx const& costMx, const unsigned int islandIdx,
                     IndividualPtrsView<GeneT>& individualPtrs) {
        if constexpr (is_noop_t<Coder>)
            // TODO: GPU
            lossFunPerIsland[islandIdx](individualPtrs, costMx,
                                        lossMx[islandIdx]);
        else {
            errors::throwNotImplemented();

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
    template <typename RNG_STATES>
    void
    breed_population(IndividualPtrsView<GeneT>& sortedParentPtrs,
                     RNG_STATES&                rngStates) {
        crossoverOp(sortedParentPtrs, params.nGenes, rngStates);
    }
};
