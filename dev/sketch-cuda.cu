#include "ev_alg/crossover.hxx"
#include "ev_alg_gpu/crossover.h"
#include "gpu/random.h"
#include "iter_utils.hxx"
#include <cuda/std/iterator>
#include <iostream>
#include <iterator>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

template <typename Iter>
void
print_device_iter(Iter begin, Iter end, std::ostream& out = std::cout,
                  std::string const& sep      = ", ",
                  std::string const& startStr = "[",
                  std::string const& endStr   = "]\n") {

    using T = typename cuda::std::iterator_traits<Iter>::value_type;

    out << startStr;
    thrust::copy(begin, end, std::ostream_iterator<T>(out, sep));
    out << endStr;
}

template <typename Container>
void
print_device_vector(Container const& v, std::ostream& out = std::cout,
                    std::string const& sep      = ", ",
                    std::string const& startStr = "[",
                    std::string const& endStr   = "]\n") {

    print_device_iter(v.cbegin(), v.cend(), out, sep, startStr, endStr);
}

template <typename F>
inline void
do_n(int n, F f) {
    for (int i{0}; i < n; ++i)
        f();
}

int
main() {
    CrossoverPMX2PointGPU     crossoverGpu;
    CrossoverPMX2PointAdapter crossover;

    std::vector<int> p1 = {1, 6, 7, 4, 3, 5, 2, 8, 9};
    std::vector<int> p2 = {3, 2, 8, 9, 1, 5, 4, 8, 6};
    std::vector<int> p3 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> p4 = {9, 8, 7, 6, 5, 4, 3, 2, 1};

    std::vector<int*> population = {p1.data(), p2.data(), p3.data(), p4.data()};

    const unsigned seed = 42;

    device::random::RndStateMemory<> states(population.size());
    device::random::initialize_rnd_states(states, {seed});

    const unsigned int nGenes = p1.size();

    std::default_random_engine cpuRng;
    cpuRng.seed(seed);

    do_n(3, [&cpuRng, &crossover] {
        std::vector<int> p1 = {1, 6, 7, 4, 3, 5, 2, 8, 9};
        std::vector<int> p2 = {3, 2, 8, 9, 1, 5, 4, 8, 6};
        std::vector<int> p3 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int> p4 = {9, 8, 7, 6, 5, 4, 3, 2, 1};

        const unsigned int nGenes = p1.size();

        std::vector<int*> population = {p1.data(), p2.data(), p3.data(),
                                        p4.data()};

        std::cout << "before crossover CPU:\n";
        for (auto const& p : {p1, p2, p3, p4}) {
            println_iter(p.begin(), p.end());
        }

        for (int i{0}; i < population.size() / 2; i += 2) {
            crossover(population[i], population[i] + nGenes, population[i + 1],
                      cpuRng);
        }

        std::cout << "after crossoverCpu:\n";
        for (auto solution : population) {
            println_iter(solution, solution + nGenes);
        }
    });

    do_n(3, [&] {
        std::cout << "before crossoverGpu:\n";
        for (auto const& p : {p1, p2, p3, p4}) {
            println_iter(p.begin(), p.end());
        }

        crossoverGpu(population, nGenes, states);

        // std::cout << "after crossoverGpu:\n";
        // for (auto solution : population) {
        //     println_iter(solution, solution + nGenes);
        // }
    });

    return 0;
}
