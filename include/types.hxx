#pragma once

#include <array>
#include <cstddef>

using llong = long long;
using size  = std::size_t;

using uchar  = unsigned char;
using uint   = unsigned int;
using ulong  = unsigned long;
using ullong = unsigned long long;

using cchar  = const char;
using cint   = const int;
using clong  = const long;
using cllong = const long long;

using cuchar  = const unsigned char;
using cuint   = const unsigned int;
using culong  = const unsigned long;
using cullong = const unsigned long long;

using cfloat   = const float;
using cdouble  = const double;
using cldouble = const long double;

template <typename T>
using PairOf = std::array<T, 2>;
