#include "device/errors.cuh"
#include "device/memory/allocator.cuh"
#include "device/memory/memory.cuh"
#include "device/random.cuh"
#include "device/reordering.cuh"
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
    std::cout << '[';
    thrust::copy(
        begin, end,
        std::ostream_iterator<typename IterT::value_type>(std::cout, ", "));
    std::cout << "]\n";
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
    device::random::uniform(random_numbers.begin(), random_numbers.end(), states);

    cudaDeviceSynchronize();
    states.print();
    random_numbers.print();
}

void
testRndMask() {
    device::random::RndStateMemory<> states(64);
    device::random::initialize_rnd_states(states);
    thrust::device_vector<bool> mask(states.size());

    device::random::mask(mask.begin(), mask.end(), states, 0.5);
    cudaDeviceSynchronize();

    printVec(mask);
}

void
testShuffleMasked() {
    device::random::RndStateMemory<> states(10);
    device::random::initialize_rnd_states(states);
    thrust::default_random_engine rng(0);

    thrust::device_vector<bool> mask(states.size());
    device::random::mask(mask.begin(), mask.end(), states, 0.3);
    cudaDeviceSynchronize();

    std::cout << "mask:   ";
    printVec(mask);

    thrust::device_vector<int> seq(mask.size());
    thrust::sequence(seq.begin(), seq.end());

    std::cout << "before: ";
    printVec(seq);

    device::random::shuffle_masked(seq.begin(), seq.end(), mask.begin(), rng);
    cudaDeviceSynchronize();

    std::cout << "after:  ";
    printVec(seq);
}

void
testShuffleWithProbability() {
    std::cout << "testShuffleWithProbability\n";

    device::random::RndStateMemory<> states(10);
    device::random::initialize_rnd_states(states);
    thrust::default_random_engine rng(0);

    thrust::device_vector<int> seq(states.size());
    thrust::sequence(seq.begin(), seq.end());

    std::cout << "before: ";
    printVec(seq);

    device::random::shuffle_with_prob(seq.begin(), seq.end(), 0.3, states, rng);

    std::cout << "after:  ";
    printVec(seq);
}

void
testChooseKWithourReplacement() {
    std::cout << "testChooseKWithourReplacement\n";

    device::random::RndStateMemory<> states(10);
    device::random::initialize_rnd_states(states);

    thrust::device_vector<int> seq(states.size());
    thrust::sequence(seq.begin(), seq.end());

    const int k = 3;

    thrust::device_vector<int> choices(k);

    device::random::choose_k_without_replacement(
        seq.begin(), seq.end(), choices.begin(), k, states);

    std::cout << "choices: ";
    printVec(choices);
}

int
main() {
    try {
        // testShuffleMasked();
        testShuffleWithProbability();
        testChooseKWithourReplacement();
        // testRnd();
        // testRndMask();
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
