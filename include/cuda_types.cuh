#pragma once

template <typename T>
using dRawVec = T* const;

template <typename T>
using dRawVecIn = const T*;

template <typename T>
using dRawVecOut = T* const;

template <typename T>
class dVec {
public:
    using size_t = unsigned long;

    const size_t size;
    T            data[];

    __device__ T
    operator[](const size_t i) const { return data[i]; }
    __device__ T&
    operator[](const size_t i) { return data[i]; }

    __host__ dVec() = delete;
};

template <typename T>
class dVecShared {
    __device__ dVecShared() = delete;
};

template <typename T>
using dVecOut = dVec<T>;

template <typename T>
using dVecIn = const dVec<T>;
