#pragma once
#include <limits>
#include <random>

template <typename PRNG = std::default_random_engine>
inline auto
create_rng(const unsigned int seed = 0u) {
    PRNG rng;
    rng.seed(seed);

    return rng;
}

template <typename T, typename PRNG>
[[nodiscard]] inline T
randint(T min, T max, PRNG& prng) {
    std::uniform_int_distribution<T> dist(min, max);
    return dist(prng);
}

template <typename T, typename PRNG>
[[nodiscard]] inline T
randint(T max, PRNG& prng) {
    return randint<T, PRNG>(0, max, prng);
}

template <typename T, typename PRNG>
[[nodiscard]] inline T
randint(PRNG& prng) {
    using limits = std::numeric_limits<T>;
    return randint<T, PRNG>(limits::min(), limits::max(), prng);
}

template <typename T, typename PRNG>
[[nodiscard]] inline std::pair<T, T>
randspan(const T min, const T max, PRNG& prng) {
    auto x1 = randint<T, PRNG>(min, max - 1, prng);
    auto x2 = randint<T, PRNG>(x1, max, prng);

    return {x1, x2};
}

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
