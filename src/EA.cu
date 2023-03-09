#include "cuda_raii.cuh"
#include "utils/text.hxx"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

template <std::default_initializable T>
struct TimesTwo {
    inline __device__ T operator()(T x) {
        return 2 * x;
    }
};

template <std::default_initializable T>
inline __host__ __device__ T timesTwo(T x) {
    return 2 * x;
}

void deviceRaiiDemo() {
    try {
        std::vector<float> hArr = {9, 8, 7, 6};

        const raii::DeviceArr<int>   dArr1(10, 42);
        raii::DeviceArr<int>         dArr2 = {1, 2, 3};
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

        // TODO fix type error
        std::cout << "transform_inplace *2:\n"
                  << "before: ";
        dArr2.print();
        std::cout << "after: ";
        // dArr2.transform_inplace(timesTwo<int>); // this results in CUDA invalid program counter error
        // always pass classes/structs with __device__ operator()
        dArr2.transform_inplace(TimesTwo<int>{});
        dArr2.print();

        std::cout << "transform: " << dArr2.transform(TimesTwo<int>{}).toString() << '\n';

    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        exit(1);
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        exit(1);
    }
}

int main() {
    deviceRaiiDemo();

    return 0;
}
