#pragma once
#include <algorithm>
#include <ranges>
#include <vector>

template <typename IterIn, typename IterStencil>
inline void
sort_by(IterIn begin, IterIn end, IterStencil stencil) {
    using Idx = long;

    // initialize indices
    const auto                   n = std::distance(begin, end);
    std::vector<Idx>             indices(n);
    const std::ranges::iota_view idxRange{0, n};
    std::copy(idxRange.begin(), idxRange.end(), indices.begin());

    // sort indices by stencil
    const auto compareByStencil = [&](const Idx& i1, const Idx& i2) { return stencil[i1] < stencil[i2]; };
    std::sort(indices.begin(), indices.end(), compareByStencil);

    // reorder according to correctly sorted indices
    reorder(begin, end, indices.cbegin());
}

enum class SortOrder {
    INCR,
    DECR
};

template <typename IterIn, typename IterStencil, bool ReorderStencil = false, SortOrder Order = SortOrder::INCR>
inline void
sort_by2(IterIn begin, IterIn end, IterStencil stencil) {
    using T           = IterIn::value_type;
    using S           = IterStencil::value_type;
    using WithStencil = std::pair<T, S>;

    // initialize indices
    const auto n = std::distance(begin, end);

    // vector of items paired with stencil elements
    std::vector<WithStencil> pairs;
    pairs.reserve(n);

    for (long i{0}; i < n; ++i)
        pairs.emplace_back(begin[i], stencil[i]);

    // sort pair by stencil
    const auto compareByStencil = [&](const WithStencil& x, const WithStencil& y) {
        if constexpr (Order == SortOrder::INCR)
            return x.second < y.second;
        else
            return x.second > y.second;
    };

    std::sort(pairs.begin(), pairs.end(), compareByStencil);

    // extract data in correct order
    if constexpr (not ReorderStencil)
        std::transform(pairs.cbegin(), pairs.cend(), begin, [](const WithStencil& paired) { return paired.first; });
    else
        for (std::size_t i{0}; i < pairs.size(); ++i) {
            const auto p = pairs[i];
            begin[i]     = p.first;
            stencil[i]   = p.second;
        }
}
