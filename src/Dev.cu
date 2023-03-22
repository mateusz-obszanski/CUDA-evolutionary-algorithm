#include "device/errors.cuh"
#include "device/memory/allocator.cuh"
#include "device/memory/memory.cuh"
#include "device/random.cuh"
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <curand_kernel.h>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

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

template <typename IterT>
inline void
printIter(IterT begin, IterT end) {
    thrust::copy(
        begin, end,
        std::ostream_iterator<typename IterT::value_type>(std::cout, ", "));
}

template <typename V>
inline void
printVec(const V& v) {
    printIter(v.begin(), v.end());
}

template <typename IterT>
inline void
PrintVecN(const IterT begin, std::size_t n) {
    auto end = begin;
    thrust::advance(end, n);
    printIter(begin, end);
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
testDeviceMemoryClass() {
    device::memory::raii::Memory<int> mem(4);
    iterPrintDevice<<<1, mem.size()>>>(mem.crbegin(), mem.crend());
    // vvv SEGFAULT
    // thrust::fill(mem.begin(), mem.end(), 42);
    thrust::fill(mem.begin_thrust(), mem.end_thrust(), 42);
    std::cout << mem.rbegin_thrust().base() << '\n';
    // or
    thrust::fill(mem.rbegin_thrust(), mem.rend_thrust(), 2137);

    auto newMem = mem.copy<float>();

    iterPrintDevice<<<1, mem.size()>>>(mem.cbegin(), mem.cend());

    mem.print();
    newMem.print();
}

template <typename State>
__global__ void
skipahead_kernel(unsigned long long n, State* state) {
    // for cuRAND
    curand_skipahead(n, state);
}

template <typename DevIter>
inline thrust::device_vector<typename DevIter::value_type>
makeCopy(DevIter begin, DevIter end) {
    thrust::device_vector<typename DevIter::value_type> xs(thrust::distance(begin, end));
    thrust::copy(begin, end, xs.begin());

    return xs;
}

template <typename DevIter>
inline thrust::device_vector<typename DevIter::value_type>
makeCopyN(DevIter begin, std::size_t n) {
    thrust::device_vector<typename DevIter::value_type> xs(n);
    thrust::copy_n(begin, n, xs.begin());

    return xs;
}

template <typename T = int>
inline thrust::device_vector<T>
makeSequence(std::size_t n) {
    thrust::device_vector<T> result(n);
    thrust::sequence(result.begin(), result.end());

    return result;
}

void
testRnd() {
    constexpr int                    N = 64;
    device::random::RndStateMemory<> states(N);
    device::random::initialize_rnd_states(states);
    device::memory::raii::DeviceMemory<float> random_numbers(N);
    device::random::generate_uniform(random_numbers.begin(), random_numbers.end(), states);

    cudaDeviceSynchronize();
    random_numbers.print();
}

void
testChooseK() {
    // auto seq = makeSequence(32);
    // printVec(chosen);
}

int
main() {
    try {
        testRnd();
        testChooseK();
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
