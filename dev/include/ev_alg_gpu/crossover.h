#pragma once
#include "gpu/errors.h"
#include "gpu/kernel_utils.h"
#include <cmath>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <curand_kernel.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

template <typename Indexable, typename IndexTransform>
struct RefIndexingAdapterGPU {
    __device__
    RefIndexingAdapterGPU(Indexable& wrapped, IndexTransform transform)
    : wrapped(wrapped), transform(transform) {}

    __device__ __forceinline__ auto&
    operator[](auto idx) {
        return wrapped[transform(idx)];
    }

    __device__ __forceinline__ auto const&
    operator[](auto idx) const {
        return wrapped[transform(idx)];
    }

private:
    Indexable&     wrapped;
    IndexTransform transform;
};

/// std::optional replacement - cuda::std does not provide it yet
enum class QuasiOptional : int { NULLOPT = -1 };

template <std::integral T>
inline __host__ __device__ constexpr bool
is_not_quasinullopt(T x) noexcept {
    return x != static_cast<T>(QuasiOptional::NULLOPT);
}

__device__ void
scan_for_duplicates_gpu(const int begin, const int end, int* const parent1,
                        int* const child, auto& mapping) {

    for (int i{begin}; i < end; ++i) {
        auto o    = mapping[parent1[i]];
        auto last = static_cast<decltype(o)>(QuasiOptional::NULLOPT);

        // look for non-duplicated value
        while (is_not_quasinullopt(o)) {
            last = o;
            o    = mapping[o];
        }

        // if was not duplicated, write from parent, else write found value
        child[i] = is_not_quasinullopt(last) ? last : parent1[i];
    }
}

__device__ void
breed_gpu(int* const p1, int* const p2, const unsigned int nGenes, int x1,
          int x2, int* const child, int* const mapping) {

    // initialize mapping with
    for (int i{0}; i < nGenes; ++i) {
        mapping[i] = static_cast<int>(QuasiOptional::NULLOPT);
    }

    const auto gene_to_index = [](auto gene) { return gene - 1; };

    // - 1, because with current encoding, numbers start from 1
    RefIndexingAdapterGPU mappingAdapter(mapping, gene_to_index);

    // Inheritance! This sets the values within the crossover zone.
    for (int i{x1}; i < x2; ++i) {
        const auto p2i = p2[i];
        child[i]       = p2i;

        // Put the values that "should have been there" into the map.
        mappingAdapter[p2i] = p1[i];
    }

    // left of the crossover zone
    scan_for_duplicates_gpu(0, x1, p1, child, mappingAdapter);
    // right -||-
    scan_for_duplicates_gpu(x2, nGenes, p1, child, mappingAdapter);
}

/// returns int in range [min, max)
template <typename CURAND_RNG_STATE>
__device__ int
randint_gpu(int min, int max, CURAND_RNG_STATE& state) {
    // due to truncating floats, randFloat * x produces [0, max - 1)
    return min + curand_uniform(&state) * (max - min);
}

template <typename CURAND_RNG_STATE>
__device__ cuda::std::pair<int, int>
           randspan_gpu(int max, CURAND_RNG_STATE& state) {

    const auto x1 = randint_gpu(0, max, state);          // x1 in [0, max - 1]
    const auto x2 = randint_gpu(x1 + 1, max + 1, state); // x2 in [x1 + 1, max]

    return {x1, x2};
}

template <typename CURAND_RNG_STATES>
__global__ void
gen_crosspoints_2(int* x1s, int* x2s, const unsigned int nCrossPointPairs,
                  const unsigned int nGenes, CURAND_RNG_STATES* states) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nCrossPointPairs)
        return;

    const auto [x1, x2] = randspan_gpu(nGenes, states[tid]);

    x1s[tid] = x1;
    x2s[tid] = x2;
}

/// threads cooperate in pairs
/// buffer, because PMX cannot be done in-place, later data will be copied back
/// to population mappingBuffer - the same size as population data population -
/// array of pointers to the beginnings of each solution
__global__ void
crossover_pmx_2p_kernel(int* const         population,
                        const unsigned int populationSize, int* const x1s,
                        int* const x2s, const unsigned int nGenes,
                        int* const childrenBuffer, int* mappingBuffer) {

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= populationSize)
        return;

    const unsigned int isThreadOdd  = tid & 0b1;
    const auto         leftParentId = tid - isThreadOdd;

    const auto p1Idx = leftParentId * nGenes;
    const auto p2Idx = p1Idx + nGenes;

    int* p1     = &population[p1Idx];
    int* p2     = &population[p2Idx];
    int* child1 = &childrenBuffer[p1Idx];
    int* child2 = &childrenBuffer[p2Idx];

    // if (isThreadOdd) {
    //     cuda::std::swap(p1, p2);
    //     cuda::std::swap(child1, child2);
    // }
    // branchless version of the above
    int* ps[]       = {p1, p2};
    int* children[] = {child1, child2};

    p1         = ps[isThreadOdd];
    p2         = ps[!isThreadOdd];
    int* child = children[isThreadOdd];

    const unsigned int crossPointIdx = tid / 2;

    int* const mapping = &mappingBuffer[tid * nGenes];

    const auto x1 = x1s[crossPointIdx];
    const auto x2 = x2s[crossPointIdx];

    // parents and child addresses have been properly swapped
    breed_gpu(p1, p2, nGenes, x1, x2, child, mapping);
}

struct CrossoverPMX2PointGPU {
    using GeneT       = int;
    using SolutionPtr = GeneT*;

    template <typename RNG_STATES>
    void
    operator()(std::span<SolutionPtr> population, const unsigned int nGenes,
               RNG_STATES& rngStates) {

        using device::kernel::utils::divCeil;

        constexpr std::size_t WARP_SIZE = 32;

        const auto nIndividuals       = population.size();
        const auto nGenesInPopulation = nIndividuals * nGenes;

        // 2 for each pair of individuals
        thrust::device_vector<int> crosspointBuff(nIndividuals);

        const auto nCrossPointPairs = nIndividuals;
        const auto nCrossPoints     = nCrossPointPairs / 2;

        int* x1s = thrust::raw_pointer_cast(crosspointBuff.data());
        int* x2s = x1s + nCrossPoints;

        const auto nBlocksCrossPoints =
            divCeil<unsigned int>(nCrossPoints, WARP_SIZE);

        gen_crosspoints_2<<<nBlocksCrossPoints, nCrossPoints>>>(
            x1s, x2s, nCrossPointPairs, nGenes, rngStates.data());

        thrust::device_vector<GeneT> d_population(nGenesInPopulation);
        thrust::device_vector<GeneT> childrenBuff(nGenesInPopulation);
        thrust::device_vector<GeneT> mappingBuffer(nGenesInPopulation);

        population_to_device(population, d_population, nIndividuals, nGenes);

        const auto nBlocks = divCeil<unsigned int>(nIndividuals, WARP_SIZE);

        crossover_pmx_2p_kernel<<<nBlocks, WARP_SIZE>>>(
            thrust::raw_pointer_cast(d_population.data()), nIndividuals, x1s,
            x2s, nGenes, thrust::raw_pointer_cast(childrenBuff.data()),
            thrust::raw_pointer_cast(mappingBuffer.data()));

        device::errors::check();

        results_to_host(population, childrenBuff, nIndividuals, nGenes);
    }

    void
    population_to_device(auto& population, auto& d_population,
                         auto nIndividuals, auto nGenes) {
        // GPU needs continuous array, so copy from host pointer locations to
        // device
        for (std::size_t i{0}; i < nIndividuals; ++i) {
            const auto solutionPtr = population[i];
            thrust::copy(solutionPtr, solutionPtr + nGenes,
                         &d_population[i * nGenes]);
        }
    }

    void
    results_to_host(std::span<SolutionPtr> population, auto& childrenBuff,
                    auto nIndividuals, auto nGenes) {
        // replace parents with offspring
        // for each pointer, overwrite its underlying memory
        for (std::size_t i{0}; i < nIndividuals; ++i) {
            auto solutionPtr = population[i];
            auto childPtr    = &childrenBuff[i * nGenes];
            thrust::copy(childPtr, childPtr + nGenes, solutionPtr);
            device::errors::check();
        }
    }
};
