#include "ev_alg/crossover.hxx"
#include "ev_alg/loss.hxx"
#include "ev_alg/migration.hxx"
#include "ev_alg/parameters.hxx"
#include "ev_alg/population_generation.hxx"
#include "ev_alg/stop_cond.hxx"
#include "ev_alg_gpu/crossover.h"
#include "ev_alg_gpu/ev_alg_gpu.h"
#include "ev_alg_gpu/mutation.h"
#include "iter_utils.hxx"
#include "perf_measure.hxx"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

[[nodiscard]] inline auto
build_island_ev_alg_tsp_pmx_gpu(AllParams const& allParams) {
    using LocationIdxT = int;
    // this might differ for special solution coder
    using GeneT = LocationIdxT;

    static_assert(std::same_as<GeneT, int>, "non-int genes not supported");
    RndPopulationGeneratorOTSP populationGenerator{allParams.islandPopulation,
                                                   allParams.nLocations};
    MutatorGPU mutator(allParams.nGenes, allParams.mutationChance);
    // CrossoverPMX2PointAdapter crossover{};
    CrossoverPMX2PointGPU crossover{};
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

    return IslandEvolutionaryAlgorithmGPU(
        populationGenerator, mutator, crossover, lossFn, migrator,
        stopCondition, params, repairOp, coder);
}

template <typename Algo = decltype(std::function{
              build_island_ev_alg_tsp_pmx_gpu})::result_type>
void
run_ev_alg_tsp_from_file_cfg_gpu(
    std::filesystem::path cfgFilePath, std::filesystem::path resultsDir,
    Algo (*builder)(AllParams const&) = build_island_ev_alg_tsp_pmx_gpu) {

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

    device::random::RndStateMemory<> states(allParams.nGenes *
                                            allParams.islandPopulation);
    device::random::initialize_rnd_states(states);
    thrust::default_random_engine thrustPrng(allParams.prng_seed);

    auto algo = builder(allParams);

    std::cout << '\n';
    std::cout << "Starting algorithm\n";

    const auto tdelta =
        measure_execution_time([&] { algo(costMx, prng, thrustPrng, states); });

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

    std::cout << "Saving results to directory " << resultsDir << "\n";

    // restore flags
    std::cout.flags(coutFlags);

    algo.save_results(resultsDir);
    allParams.save(resultsDir / "parameters.txt");
}

/// Arguments:
/// [cfgFilePath] [resultsDir]
int
main(const int argc, const char* const argv[]) {
    try {
        std::filesystem::path cfgFilePath, resultsDir;

        if (argc < 2)
            cfgFilePath = "algo_cfg.txt";
        else
            cfgFilePath = argv[1];

        if (argc < 3)
            resultsDir = std::filesystem::current_path() / "results-gpu" /
                         get_time_str("%d-%m-%Y_%H-%M-%S");
        else
            resultsDir = argv[2];

        run_ev_alg_tsp_from_file_cfg_gpu(cfgFilePath, resultsDir);
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        std::exit(1);
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        throw;
    }
    return 0;
}

// int
// main() {
//     constexpr unsigned N_GENES         = 10;
//     constexpr unsigned THRUST_RNG_SEED = 42;
//     constexpr float    MUTATION_PROB   = 0.8f;
//
//     device::random::RndStateMemory<> states(N_GENES);
//     device::random::initialize_rnd_states(states);
//     thrust::default_random_engine rng(THRUST_RNG_SEED);
//
//     thrust::device_vector<int> genome(states.size());
//     thrust::sequence(genome.begin(), genome.end());
//
//     MutatorGPU mutator(genome.size(), MUTATION_PROB);
//     mutator(genome.begin(), rng, states);
//
//     std::cout << "after:\n";
//     thrust::host_vector<int> result(genome);
//     println_iter(result.cbegin(), result.cend());
//
//     return 0;
// }
