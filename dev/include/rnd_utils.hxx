#pragma once
#include <random>

inline auto
create_rng(const unsigned int seed = 0u) {
    std::default_random_engine rng;
    rng.seed(seed);

    return rng;
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
    return 1 - 2 * getRndBit(prng);
}
