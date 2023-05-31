#include "algo_utils.hxx"
#include "ea_utils.hxx"
#include "iter_utils.hxx"
#include "permutation.hxx"
#include "rnd_utils.hxx"
#include "string_utils.hxx"
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

void
playground() {
    // CPU IMPLEMENTATION
    constexpr unsigned int PRNG_SEED       = 100;
    constexpr int          POPULATION_SIZE = 4;
    constexpr int          N_LOCATIONS     = 8; // n points for TSP
    constexpr float        MUTATION_CHANCE = 0.5;
    constexpr int          N_POPULATIONS   = 3;
    constexpr int          N_MIGRANTS      = 2;
    // constexpr float MIGRATION_CHANCE = 0.25;
    constexpr auto N_GENES = N_LOCATIONS - 1;

    auto prng = create_rng(PRNG_SEED);

    // TEST MUTATION
    std::vector<int> genome;
    genome.reserve(16);
    for (int i{0}; i < 16; ++i) {
        genome.push_back(i);
    }

    // preallocate helper vector
    // use char instead of bool, because vector<bool> is bad
    std::vector<char> mask(N_LOCATIONS);

    choice_shuffle(MUTATION_CHANCE, genome.begin(), genome.end(), prng, mask);

    std::cout << "genome: ";
    printContainer(genome);

    const auto spawnRndPopulation = [=, &prng]() { return create_rnd_solution_population_tsp<int>(POPULATION_SIZE, N_LOCATIONS, prng); };

    auto population = spawnRndPopulation();

    const auto createDummySolution = [] {
        std::vector<int> dummySolution;
        dummySolution.reserve(POPULATION_SIZE * (N_LOCATIONS - 1));
        for (int i{0}; i < POPULATION_SIZE; ++i)
            for (int j{1}; j < N_LOCATIONS; ++j)
                dummySolution.push_back(j);

        return dummySolution;
    };

    const auto createDummySolution2 = [] {
        std::vector<int> dummySolution;
        dummySolution.reserve(POPULATION_SIZE * (N_LOCATIONS - 1));
        for (int i{1}; i < POPULATION_SIZE * (N_LOCATIONS - 1); ++i)
            dummySolution.push_back(i);

        return dummySolution;
    };

    auto dummySolution = createDummySolution();

    const auto printDummySolution = [&] { prettyPrintMx(dummySolution.cbegin(), POPULATION_SIZE, N_GENES); };

    std::cout << "whole dummy population before mutation:\n";
    printDummySolution();

    std::vector<char> solutionMask(POPULATION_SIZE * N_LOCATIONS);
    mutatePopulation(dummySolution.begin(), POPULATION_SIZE, N_GENES, MUTATION_CHANCE, prng, solutionMask);

    std::cout << "after mutation:\n";
    printDummySolution();

    // TEST RANDOM COST MATRIX
    const auto costMx = create_rnd_symmetric_cost_mx_tsp(N_LOCATIONS, prng);

    std::cout << "Cost matrix:\n";
    prettyPrintMx(costMx.begin(), N_LOCATIONS);

    // TEST LOSS FUNCTION
    // create vector of pointers - individuals
    auto ind1 = solution_to_individuals(dummySolution, POPULATION_SIZE, N_GENES);

    std::vector<float> losses(POPULATION_SIZE);
    std::transform(
        ind1.cbegin(), ind1.cend(), losses.begin(),
        [&](auto ind) { return calcLossClosed(costMx, N_LOCATIONS, ind, ind + N_GENES); });

    std::cout << "losses: ";
    printContainer(losses);

    // TEST MIGRATION
    auto s1 = spawnRndPopulation();
    auto s2 = spawnRndPopulation();

    std::cout << "population1:\n";
    prettyPrintMx(s1.begin(), POPULATION_SIZE, N_GENES);
    std::cout << "population2:\n";
    prettyPrintMx(s2.begin(), POPULATION_SIZE, N_GENES);

    auto p1 = solution_to_individuals(s1, POPULATION_SIZE, N_GENES);
    auto p2 = solution_to_individuals(s2, POPULATION_SIZE, N_GENES);

    auto losses1 = calcPopulationLosses(p1, costMx, N_LOCATIONS);
    auto losses2 = calcPopulationLosses(p2, costMx, N_LOCATIONS);

    std::cout << "population1 losses: ";
    printContainer(losses1);
    std::cout << "population2 losses: ";
    printContainer(losses2);

    std::cout << "no. of migrants: " << N_MIGRANTS << '\n';

    // FIXME: sort_by2 p<k> and losses<k> before migrateBetween
    using PopIter               = decltype(p1.begin());
    using LossIter              = decltype(losses1.begin());
    constexpr bool REORDER_LOSS = true;
    sort_by2<PopIter, LossIter, REORDER_LOSS, SortOrder::DECR>(p1.begin(), p1.begin() + POPULATION_SIZE, losses1.begin());
    sort_by2<PopIter, LossIter, REORDER_LOSS, SortOrder::DECR>(p2.begin(), p2.begin() + POPULATION_SIZE, losses2.begin());
    migrateBetween(p1, p2, N_GENES, losses1, losses2, N_MIGRANTS);

    std::cout << "after migration:\n";
    std::cout << "population1:\n";
    prettyPrintMx(s1.begin(), POPULATION_SIZE, N_GENES);
    std::cout << "population2:\n";
    prettyPrintMx(s2.begin(), POPULATION_SIZE, N_GENES);

    // test full migration
    using Gene          = int;
    using RawPopulation = std::vector<Gene>;
    using PopulationVec = std::vector<RawPopulation>;

    PopulationVec populations;

    for (int i{0}; i < N_POPULATIONS; ++i)
        populations.push_back(spawnRndPopulation());

    using IndividualVec               = std::vector<Individual<Gene>>;
    using PopulationsAsIndividualsVec = std::vector<IndividualVec>;

    // necessary for migration and loss calculation
    PopulationsAsIndividualsVec populationsAsIndividuals(N_POPULATIONS);
    std::transform(
        populations.begin(), populations.end(),
        populationsAsIndividuals.begin(),
        [](auto& p) { return std::move(solution_to_individuals(p, POPULATION_SIZE, N_GENES)); });

    using PopulationLossVec     = std::vector<float>;
    using PopulationsLossMatrix = std::vector<PopulationLossVec>;
    PopulationsLossMatrix popLossMx(N_POPULATIONS);
    std::transform(
        populationsAsIndividuals.cbegin(), populationsAsIndividuals.cend(),
        popLossMx.begin(),
        [&costMx](const auto& p) { return std::move(calcPopulationLosses(p, costMx, N_LOCATIONS)); });

    std::cout << "full migration test:\n";
    std::cout << "populations:\n";

    for (const auto& p : populations) {
        printSeparationLine(3);
        prettyPrintMx(p.begin(), POPULATION_SIZE, N_GENES);
    }

    std::cout << "loss function matrix (row - population, column - individual):\n";
    prettyPrintMx(popLossMx);

    const auto migrationDirection = getRandomMigrationDirection(prng); // +/- 1

    std::cout << "migration direction: " << (migrationDirection == MigrationDirection::LEFT ? "LEFT" : "RIGHT") << '\n';

    migrate(populationsAsIndividuals, popLossMx, POPULATION_SIZE, N_GENES, N_MIGRANTS, migrationDirection);

    std::cout << "populations after full migration:\n";

    for (const auto& p : populations) {
        printSeparationLine(3);
        prettyPrintMx(p.begin(), POPULATION_SIZE, N_GENES);
    }

    std::cout << "loss function matrix (row - population, column - individual):\n";
    prettyPrintMx(popLossMx);
}

inline void
showcase_permutation_inversion_sequence() {
    const auto check_ok = [](const auto& a, const auto& b) {
        std::cout << "ok? " << yes_no(a == b) << "\n\n";
    };

    // std::vector<int> inversion_vec_gt{3, 2, 1, 0, 0};
    // std::vector<int> permutation_gt{3, 2, 1, 0, 4};
    std::vector<int> inversion_vec_gt{1, 1, 2, 1, 0};
    std::vector<int> permutation_gt{4, 0, 1, 3, 2};

    using Iter            = decltype(inversion_vec_gt.cbegin());
    constexpr auto Coding = InversionVectorCoding::SAME;

    const auto permutation = inversion_vector_to_permutation<Iter, Coding>(inversion_vec_gt.cbegin(), inversion_vec_gt.size());

    std::cout << "permutation ground-truth: ";
    printlnContainer(permutation_gt);

    std::cout << "permutation: ";
    printlnContainer(permutation);

    check_ok(permutation, permutation_gt);

    const auto inversion_vec = permutation_to_inversion_vector<Iter, Coding>(permutation_gt.cbegin(), permutation_gt.size());

    std::cout << "inversion vector ground-truth: ";
    printlnContainer(inversion_vec_gt);

    std::cout << "to inversion vector: ";
    printlnContainer(inversion_vec);

    check_ok(inversion_vec, inversion_vec_gt);

    std::cout << "abstract coders:\n";
    std::vector<std::unique_ptr<AbstractPermutationCoder>> coders;
    coders.push_back(std::make_unique<PolymorphicPermutationCoderSame>());
    coders.push_back(std::make_unique<PolymorphicPermutationCoderShort>());

    for (const auto& coder : coders) {
        std::cout << "coder: " << coder->get_coding_str() << '\n';

        std::cout << "permutation ground-truth: ";
        printlnContainer(permutation_gt);

        std::cout << "inversion vector ground-truth: ";
        printlnContainer(inversion_vec_gt);

        auto inv = coder->encode(permutation_gt);

        // for SHORT coding, to check we must append 0
        if (coder->get_coding() == InversionVectorCoding::SHORT)
            inv.push_back(0);

        std::cout << "encoded: ";
        printlnContainer(inv);

        check_ok(inv, inversion_vec_gt);

        auto inversion_vec_gt2 = inversion_vec_gt;

        if (coder->get_coding() == InversionVectorCoding::SHORT)
            inversion_vec_gt2.pop_back();

        const auto perm = coder->decode(inversion_vec_gt2);

        std::cout << "decoded: ";
        printlnContainer(perm);

        check_ok(perm, permutation_gt);

        auto original_perm = permutation_gt;

        std::cout << "test perm->encode->decode:";
        const auto undone1 = coder->decode(coder->encode(permutation_gt));
        std::cout << "after: ";
        printlnContainer(undone1);
        check_ok(undone1, permutation_gt);

        std::cout << "test perm->decode->encode:";
        const auto undone2 = coder->encode(coder->decode(inversion_vec_gt2));
        std::cout << "after: ";
        printlnContainer(undone2);
        check_ok(undone2, inversion_vec_gt2);
    }
}

int
main() {
    try {
        // playground();
        showcase_permutation_inversion_sequence();
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        return 1;
    }
    return 0;
}
