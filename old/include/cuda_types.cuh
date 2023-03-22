#pragma once

template <typename T>
using device_ptr = T* const;

template <typename T>
using host_ptr = T* const;

template <typename T>
using device_ptr_in = const T*;

template <typename T>
using device_ptr_out = T* const;

template <typename T>
class device_arr {
public:
    using size_t = unsigned long;

    const size_t size;
    T            data[];

    __device__ T
    operator[](const size_t i) const {
        return data[i];
    }
    __device__ T&
    operator[](const size_t i) {
        return data[i];
    }

    __host__
    device_arr() = delete;
};

template <typename T>
class device_arr_shared {
    __device__
    device_arr_shared() = delete;
};

template <typename T>
using device_arr_out = device_arr<T>;

template <typename T>
using device_arr_in = const device_arr<T>;
