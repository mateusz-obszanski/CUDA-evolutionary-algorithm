#include "device/errors.cuh"
#include "device/memory/allocator.cuh"
#include "device/memory/memory.cuh"
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <curand_kernel.h>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

__global__ void
mul2(int* out, cuda::std::size_t len) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= len)
        return;

    out[tid] *= 2;
}

template <typename Iter>
__global__ void
iterPrintDevice(Iter begin, Iter end) {
    const auto diff = end - begin;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= std::abs(diff))
        return;

    const auto p = begin + tid;

    printf("printf from device: %d\n", *p);
}

__global__ void
rndGen2() {
    using rnd_params_t = unsigned long long;

    const rnd_params_t seed     = 0;
    const rnd_params_t sequence = 0;
    const rnd_params_t offset   = 0;

    curandState_t state;
    curand_init(seed, sequence, offset, &state);

    const auto x1 = curand_uniform(&state);
    const auto x2 = curand_uniform(&state);
}

void
testManagedMemory() {
    constexpr cuda::std::size_t N = 5;

    int* arr;

    cudaMallocManaged(&arr, N * sizeof(int));

    for (size_t i{0}; i < N; ++i)
        arr[i] = i;

    mul2<<<1, N>>>(arr, N);
    if (const auto err = cudaGetLastError())
        std::cout << "GPU ERROR: " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << '\n';

    thrust::copy(arr, arr + N, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << '\n';

    cudaFree(arr);
}

void
testRnd() {
    rndGen2<<<1, 1>>>();
    device::errors::check();

    cudaDeviceSynchronize();
}

int
main() {
    device::memory::raii::DeviceMemory<int> mem(4);
    std::cout << 1 << '\n';
    iterPrintDevice<<<1, mem.size()>>>(mem.crbegin(), mem.crend());
    // vvv SEGFAULT
    // thrust::fill(mem.begin(), mem.end(), 42);
    thrust::fill(mem.begin_thrust(), mem.end_thrust(), 42);
    std::cout << 2 << '\n';
    std::cout << mem.rbegin_thrust().base() << '\n';
    // or
    thrust::fill(mem.rbegin_thrust(), mem.rend_thrust(), 2137);
    std::cout << 3 << '\n';

    auto newMem = mem.copy<float>();
    std::cout << 4 << '\n';

    iterPrintDevice<<<1, mem.size()>>>(mem.cbegin(), mem.cend());
    std::cout << 5 << '\n';

    mem.print();
    std::cout << 6 << '\n';
    newMem.print();
    std::cout << 7 << '\n';

    return 0;
}
