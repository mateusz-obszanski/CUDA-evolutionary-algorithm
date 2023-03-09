#include <iostream>

int
main() {
    cudaDeviceProp props;
    const int      device = 0;
    cudaGetDeviceProperties(&props, device);

    std::cout << "version: " << props.major << '.' << props.minor << '\n';
}
