#include "cuda_raii.cuh"
#include "utils/text.hxx"
#include <iostream>

int main() {
    std::cout << "Hello world!" << '\n';

    const raii::DeviceArr dArr(10, 0);
    utils::printVec(dArr.toHost());

    return 0;
}
