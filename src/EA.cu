#include "cuda_raii.cuh"
#include "utils/text.hxx"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

template <std::default_initializable T>
struct TimesTwo {
    inline __device__ T
    operator()(T x) {
        return 2 * x;
    }
};

template <std::default_initializable T, concepts::MappingFn<T> F>
struct DeviceLambda {
private:
    F f;

public:
    DeviceLambda(F f) : f(f) {}
    inline __device__ T
    operator()(const T x) {
        return f(x);
    }
};

template <std::default_initializable T>
inline __host__ __device__ T
timesTwo(T x) {
    return 2 * x;
}

void
deviceRaiiDemo() {
    try {
        std::vector<float> hArr = {9, 8, 7, 6};

        const raii::DeviceArr<int>   dArr1(10, 42);
        const raii::DeviceArr<int>   dArr2 = {1, 2, 3};
        const raii::DeviceArr<float> dArr3{hArr.begin(), hArr.end()};
        const raii::DeviceArr<float> dArr4{hArr};
        const raii::DeviceArr<float> dArr5{std::array<float, 3>{4.5, 5, 6}};
        const raii::DeviceArr<float> dArr6{hArr.data(), hArr.size()};

        const auto arr1back = dArr1.toHost();
        utils::printVec(arr1back);

        dArr1.print();
        dArr2.print();
        dArr3.print();
        dArr4.print();
        dArr5.print();
        dArr6.print();

        // changing copy type
        raii::DeviceArr<float> dArr2copy = dArr2.copy<float>();

        // TODO fix type error
        std::cout << "transform_inplace *2:\n"
                  << "before: ";
        dArr2copy.print();
        std::cout << "after: ";
        // dArr2copy.transform_inplace(timesTwo<int>); // this results in CUDA invalid program counter error
        // always pass classes/structs with __device__ operator()
        dArr2copy.transform_inplace(TimesTwo<int>{});
        dArr2copy.print();

        std::cout << "transform: " << dArr2copy.transform(TimesTwo<int>{}).toString() << '\n';
        // vvv not working
        // std::cout << "lambda transform: " << dArr2.transform(DeviceLambda([](const int x) -> int { return 2 * x; })).toString();
        // vvv this works only with --extended-lambda NVCC compilation flag
        std::cout << "lambda transform: " << dArr2copy.transform([] __device__(const int x) -> int { return 2 * x; }).toString() << '\n';
        std::cout << "partial transform: " << dArr2copy.transform(cuda_ops::Partial<cuda_ops::Mult<float>, float>(2)).toString() << '\n';
        std::cout << "sum 10 000 ones: " << raii::DeviceArr<std::size_t>::createOnes(10'000).reduce() << '\n';
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        exit(1);
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        exit(1);
    }
}

int
main() {
    deviceRaiiDemo();

    return 0;
}
