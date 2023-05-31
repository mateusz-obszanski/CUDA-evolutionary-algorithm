#pragma once
#include "./iter_utils.hxx"
#include <vector>

/// assumption: perm elements in {0, 1, ..., n - 1}
template <typename IterPerm>
inline std::vector<int>
permutation_to_inversion_vector(IterPerm perm, const int n) {
    std::vector<int> inv(n);

    for (int i{0}; i < n; ++i) {
        int  m      = 0;
        auto perm_m = perm[m];

        while (perm_m != i) {
            // branchless increment if condition is true
            inv[i] += perm_m > i;
            perm_m = perm[++m];
        }
    }

    return inv;
}

template <typename IterInv>
[[nodiscard]] inline std::vector<int>
inversion_vector_to_permutation(IterInv inv, const int n) {
    auto             indices = range_vec(n);
    std::vector<int> permutation;
    permutation.reserve(n);

    for (int i{0}; i < n; ++i) {
        const auto idx = inv[i];
        permutation.push_back(indices[idx]);
        indices.erase(indices.begin() + idx);
    }

    return permutation;
}

/// inv[i] must be < n - 1 - i (only n - 1 - i elements on left can be
/// greater than i in permutation)
template <typename Iter>
inline void
repair_inversion_vector(Iter begin, Iter end) {
    using T = typename Iter::value_type;

    const T n = std::distance(begin, end);
    std::transform(begin, end, begin, [=](const auto& x) { return x % n; });
}
