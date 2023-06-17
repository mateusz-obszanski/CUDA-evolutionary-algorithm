#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ void
load_cost_mx_to_shared(unsigned const tid, unsigned const threadStride,
                       float* const s_costMx, float const* const costMx,
                       unsigned const nLocations) {
    // Load costMx to shared memory
    const auto costMxSize = nLocations * nLocations;

    for (auto i{tid}; i < costMxSize; i += threadStride) {
        s_costMx[tid] = costMx[tid];
    }
}

__device__ inline void
dev_calc_losses_open(float* const losses, int const** const solutions,
                     unsigned const nSolutions, unsigned const nGenes,
                     float const* const costMx, unsigned const nLocations) {

    extern __shared__ float s_costMx[];

    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    load_cost_mx_to_shared(tid, blockDim.x, s_costMx, costMx, nLocations);

    if (tid >= nSolutions * nGenes)
        return;

    const bool isPrevStartingPoint = tid % nGenes == 0;

    // cost matrix indices
    const unsigned src = isPrevStartingPoint ? 0 : (*solutions)[tid - 1];
    const unsigned dst = (*solutions)[tid];

    const unsigned individualIdx = tid / nGenes;
    atomicAdd(&losses[individualIdx], costMx[src * nLocations + dst]);
}

__global__ void
calc_losses_closed(float* const losses, int const** const solutions,
                   unsigned const nSolutions, unsigned const nGenes,
                   float const* const costMx, unsigned const nLocations) {}
