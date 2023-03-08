#include "cuda_raii.cuh"
#include "utils/text.hxx"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

void deviceRaiiDemo() {
    try {
        std::vector<float> hArr = {9, 8, 7, 6};

        const raii::DeviceArr<int>   dArr1(10, 42);
        const raii::DeviceArr<int>   dArr2 = {1, 2, 3};
        const raii::DeviceArr<float> dArr3{hArr.begin(), hArr.end()};
        const raii::DeviceArr<float> dArr4{hArr};
        const raii::DeviceArr<float> dArr5{std::array<float, 3>{4.5, 5, 6}};
        const raii::DeviceArr<float> dArr6{hArr.data(), hArr.size()};

        utils::printVec(dArr1.toHost());
        utils::printVec(dArr2.toHost());
        utils::printVec(dArr3.toHost());
        utils::printVec(dArr4.toHost());
        utils::printVec(dArr5.toHost());
        utils::printVec(dArr6.toHost());
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
