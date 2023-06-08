#include "ev_alg/crossover.hxx"
#include "ev_alg/ev_alg.hxx"
#include "ev_alg/loss.hxx"
#include "ev_alg/migration.hxx"
#include "ev_alg/mutation.hxx"
#include "ev_alg/parameters.hxx"
#include "ev_alg/population_generation.hxx"
#include "ev_alg/stop_cond.hxx"
#include "utils.hxx"
#include <algorithm>
#include <cstddef>

inline void
show_ev_alg_tsp() {
    const std::size_t  prng_seed                 = 0;
    const unsigned int iterationsPerEpoch        = 10;
    const unsigned int nEpochs                   = 6;
    const unsigned int nEpochsWithourImprovement = 2;

    const unsigned int nLocations       = 4;
    const float        minCost          = 1e-1f;
    const float        maxCost          = 1e1;
    const unsigned int islandPopulation = 4; // must be an even number
    const unsigned int nIslands         = 4;
    // in normal, same-size coding, the first and last (0) location is implicit
    // if special solution coder is used, this will differ
    const unsigned int nGenes         = nLocations - 1;
    const float        mutationChance = 0.1f;
    const float        migrationRatio = 0.1f;
    const auto         nMigrants      = static_cast<unsigned int>(
        std::max(1.0f, migrationRatio * islandPopulation));

    auto       prng = create_rng(prng_seed);
    const auto costMx =
        create_rnd_symmetric_cost_mx_tsp(nLocations, prng, minCost, maxCost);
    using LocationIdxT = int;
    // this might differ for special solution coder
    using GeneT = LocationIdxT;

    static_assert(std::same_as<GeneT, int>, "non-int genes not supported");
    static_assert(isEven(islandPopulation),
                  "island population MUST be even, because parents are grouped "
                  "pairwise for breeding");

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

    std::cout << "Configuring algorithm\n";

    // PMX
    IslandEvolutionaryAlgorithm algo(populationGenerator, mutator, crossover,
                                     lossFn, migrator, stopCondition, prng,
                                     params, repairOp, coder);

    std::cout << "Starting algorithm\n";
    algo();

    std::cout << "stop reason: " << algo.stopCondition.get_stop_reason_str()
              << '\n';

    const auto resultsDir = std::filesystem::current_path() / "results/" /
                            get_time_str("%d-%m-%Y_%H-%M-%S");

    std::cout << "Saving results to directory " << resultsDir << "\n";

    algo.save_results(resultsDir);
}

void
test_param_loading() {
    const auto params = AllParams::from_file("algo_cfg.txt");
    params.print();
}

void
show_ev_alg_tsp_from_fie_cfg() {
    const auto allParams = AllParams::from_file("algo_cfg.txt");

    auto       prng   = create_rng(allParams.prng_seed);
    const auto costMx = create_rnd_symmetric_cost_mx_tsp(
        allParams.nLocations, prng, allParams.minCost, allParams.maxCost);
    using LocationIdxT = int;
    // this might differ for special solution coder
    using GeneT = LocationIdxT;

    static_assert(std::same_as<GeneT, int>, "non-int genes not supported");
    RndPopulationGeneratorOTSP populationGenerator{allParams.islandPopulation,
                                                   allParams.nLocations};
    Mutator mutator(allParams.nGenes, allParams.mutationChance);
    CrossoverPMX2PointAdapter crossover{};
    LossFnClosed              lossFn(allParams.nLocations, &costMx,
                                     allParams.islandPopulation);
    MigrationOp<GeneT> migrator(allParams.islandPopulation, allParams.nGenes,
                                allParams.nMigrants);
    StopCondition      stopCondition(allParams.nEpochs,
                                     allParams.nEpochsWithoutImprovement);
    const EvAlgParams  params(costMx, allParams.islandPopulation,
                              allParams.nLocations, allParams.nIslands,
                              allParams.iterationsPerEpoch, allParams.nGenes);
    // those are optional
    auto repairOp = NoOp();
    auto coder    = NoOp();

    std::cout << "Configuring algorithm\n"
              << "Parameters:\n";
    allParams.print();

    // PMX
    IslandEvolutionaryAlgorithm algo(populationGenerator, mutator, crossover,
                                     lossFn, migrator, stopCondition, prng,
                                     params, repairOp, coder);

    std::cout << "Starting algorithm\n";
    algo();

    std::cout << "Finished!\n"
              << "stop reason: " << algo.stopCondition.get_stop_reason_str()
              << '\n'
              << "epochs: " << algo.stopCondition.get_iters() << '/'
              << allParams.nEpochs << '\n'
              << "epochs since improvement: "
              << algo.stopCondition.get_iters_since_improvement() << '/'
              << allParams.nEpochsWithoutImprovement << '\n';

    const auto resultsDir = std::filesystem::current_path() / "results/" /
                            get_time_str("%d-%m-%Y_%H-%M-%S");

    std::cout << "Saving results to directory " << resultsDir << "\n";

    algo.save_results(resultsDir);
}

int
main() {
    try {
        // playground();
        // showcase_permutation_inversion_sequence();
        // pmx_crossover_showcase();
        // test_param_loading();
        // show_ev_alg_tsp();
        show_ev_alg_tsp_from_fie_cfg();
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        std::exit(1);
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        throw;
    }
    return 0;
}
