#pragma once
#include <chrono>
#include <functional>

using MilliSecF64 = double;

// template <typename F>
inline double
measure_execution_time(std::function<void()> f) {
    using std::chrono::high_resolution_clock;

    const auto t1 = high_resolution_clock::now();
    f();
    const auto t2 = high_resolution_clock::now();

    const std::chrono::duration<MilliSecF64, std::milli> tdelta = t2 - t1;

    return tdelta.count();
}
