#include "ev_alg/crossover.hxx"
#include "ev_alg/ev_alg.hxx"
#include "ev_alg/loss.hxx"
#include "ev_alg/migration.hxx"
#include "ev_alg/mutation.hxx"
#include "ev_alg/parameters.hxx"
#include "ev_alg/population_generation.hxx"
#include "ev_alg/stop_cond.hxx"
#include "iter_utils.hxx"
#include "utils.hxx"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <ios>

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
    LossFnClosed               lossFn(nLocations, islandPopulation);
    MigrationOp<GeneT>         migrator(islandPopulation, nGenes, nMigrants);
    StopCondition     stopCondition(nEpochs, nEpochsWithourImprovement);
    const EvAlgParams params(islandPopulation, nLocations, nIslands,
                             iterationsPerEpoch, nGenes);
    // those are optional
    auto repairOp = NoOp();
    auto coder    = NoOp();

    std::cout << "Configuring algorithm\n";

    // PMX
    IslandEvolutionaryAlgorithm algo(populationGenerator, mutator, crossover,
                                     lossFn, migrator, stopCondition, params,
                                     repairOp, coder);

    std::cout << "Starting algorithm\n";
    algo(costMx, prng);

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

[[nodiscard]] inline auto
build_island_ev_alg_tsp_pmx(AllParams const& allParams) {
    using LocationIdxT = int;
    // this might differ for special solution coder
    using GeneT = LocationIdxT;

    static_assert(std::same_as<GeneT, int>, "non-int genes not supported");
    RndPopulationGeneratorOTSP populationGenerator{allParams.islandPopulation,
                                                   allParams.nLocations};
    Mutator mutator(allParams.nGenes, allParams.mutationChance);
    CrossoverPMX2PointAdapter crossover{};
    LossFnClosed       lossFn(allParams.nLocations, allParams.islandPopulation);
    MigrationOp<GeneT> migrator(allParams.islandPopulation, allParams.nGenes,
                                allParams.nMigrants);
    StopCondition      stopCondition(allParams.nEpochs,
                                     allParams.nEpochsWithoutImprovement);
    const EvAlgParams  params(allParams.islandPopulation, allParams.nLocations,
                              allParams.nIslands, allParams.iterationsPerEpoch,
                              allParams.nGenes);
    // those are optional
    auto repairOp = NoOp();
    auto coder    = NoOp();

    return IslandEvolutionaryAlgorithm(populationGenerator, mutator, crossover,
                                       lossFn, migrator, stopCondition, params,
                                       repairOp, coder);
}

using MilliSecF64 = double;

// template <typename F>
inline double
measure_execution_time(std::function<void()> f) {
    using std::chrono::high_resolution_clock;

    const auto t1 = high_resolution_clock::now();
    f();
    const auto t2 = high_resolution_clock::now();

    const std::chrono::duration<MilliSecF64, std::milli> tdelta = t2 - t1;

    return tdelta.count();
}

template <typename Algo = decltype(std::function{
              build_island_ev_alg_tsp_pmx})::result_type>
void
show_ev_alg_tsp_from_file_cfg(
    std::filesystem::path cfgFilePath,
    Algo (*builder)(AllParams const&) = build_island_ev_alg_tsp_pmx) {

    std::cout << "Reading experiment configuration from file \"" << cfgFilePath
              << "\"\n";
    const auto allParams = AllParams::from_file(cfgFilePath);

    std::cout << "Setting random seed (" << allParams.prng_seed << ")\n";
    auto prng = create_rng(allParams.prng_seed);

    std::cout << "Generating random cost matrix\n";
    const auto costMx = create_rnd_symmetric_cost_mx_tsp(
        allParams.nLocations, prng, allParams.minCost, allParams.maxCost);

    std::cout << '\n';
    std::cout << "Configuring algorithm:\n"
              << "---\n"
              << "Parameters:\n";
    allParams.print();

    auto algo = builder(allParams);

    std::cout << '\n';
    std::cout << "Starting algorithm\n";

    const auto tdelta =
        measure_execution_time([&algo, &costMx, &prng] { algo(costMx, prng); });

    const auto coutFlags = std::cout.flags();
    std::cout << std::scientific;

    std::cout << "Finished!\n"
              << "Best solution loss: " << algo.get_best_fitness() << '\n'
              << "Elapsed time: " << tdelta << "[ms]\n"
              << "stop reason: " << algo.stopCondition.get_stop_reason_str()
              << '\n'
              << "epochs: " << algo.stopCondition.get_iters() << '/'
              << allParams.nEpochs << '\n'
              << "epochs since improvement: "
              << algo.stopCondition.get_iters_since_improvement() << '/'
              << allParams.nEpochsWithoutImprovement << '\n';

    std::cout << "Best solution:\n";
    println_container(algo.get_best_solution());

    const auto resultsDir = std::filesystem::current_path() / "results/" /
                            get_time_str("%d-%m-%Y_%H-%M-%S");

    std::cout << "Saving results to directory " << resultsDir << "\n";

    // restore flags
    std::cout.flags(coutFlags);

    algo.save_results(resultsDir);
    allParams.save(resultsDir / "parameters.txt");
}

int
main(const int argc, const char* const argv[]) {
    try {
        // playground();
        // showcase_permutation_inversion_sequence();
        // pmx_crossover_showcase();
        // test_param_loading();
        // show_ev_alg_tsp();
        std::filesystem::path cfgFilePath;

        if (argc < 2)
            cfgFilePath = "algo_cfg.txt";
        else
            cfgFilePath = argv[1];

        show_ev_alg_tsp_from_file_cfg(cfgFilePath);
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        std::exit(1);
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        throw;
    }
    return 0;
}
