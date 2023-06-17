#pragma once
#include <limits>
#include <random>

template <typename PRNG = std::mt19937>
inline auto
create_rng(const unsigned long seed = 0u) {
    PRNG rng;
    rng.seed(seed);

    return rng;
}

/// [min, max]
template <typename T, typename PRNG>
[[nodiscard]] inline T
randint(T min, T max, PRNG& prng) {
    std::uniform_int_distribution<T> dist(min, max);
    return dist(prng);
}

/// [min, max]
template <typename T, typename PRNG>
[[nodiscard]] inline T
randint(T max, PRNG& prng) {
    return randint<T, PRNG>(0, max, prng);
}

/// [min, max]
template <typename T, typename PRNG>
[[nodiscard]] inline T
randint(PRNG& prng) {
    using limits = std::numeric_limits<T>;
    return randint<T, PRNG>(limits::min(), limits::max(), prng);
}

/// min < max
/// @returns x1, x2 in [min, max], x1 < x2
template <typename T, typename PRNG>
[[nodiscard]] inline std::pair<T, T>
randspan(const T min, const T max, PRNG& prng) {
    auto x1 = randint<T, PRNG>(min, max - 1, prng);
    auto x2 = randint<T, PRNG>(x1 + 1, max, prng);

    return {x1, x2};
}

/// min < max
/// @returns x1, x2 in [min, max], x1 < x2
template <typename T, typename PRNG>
[[nodiscard]] inline std::pair<T, T>
randspan(const T max, PRNG& prng) {
    return randspan<T, PRNG>(0, max, prng);
}

template <typename PRNG>
inline int
getRndBit(PRNG& prng) {
    std::uniform_int_distribution binaryDist(0, 1);
    return binaryDist(prng);
}

template <typename PRNG>
inline int
getRandomSign(PRNG& prng) {
    return 1 - 2 * getRndBit<PRNG>(prng);
}
