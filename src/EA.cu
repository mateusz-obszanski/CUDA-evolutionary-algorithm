#include "cuda_raii.cuh"
#include "utils/text.hxx"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        std::cout << "Hello world!" << '\n';

        const raii::DeviceArr<int> dArr(10, 42);

        utils::printVec(dArr.toHost());

        const raii::DeviceArr<int> dArr2 = {1, 2, 3};

        utils::printVec(dArr2.toHost());
    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN ERROR!\n";
        return 1;
    }

    return 0;
}
