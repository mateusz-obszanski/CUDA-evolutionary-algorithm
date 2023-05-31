#pragma once
#include <algorithm>
#include <concepts>
#include <random>
#include <ranges>
#include <vector>

template <
    typename InIter, typename OutIter,
    typename StencilIter,
    std::predicate<typename StencilIter::value_type> PredicateT>
inline void
gather_if(
    InIter begin, InIter end,
    OutIter result, StencilIter stencil, PredicateT predicate) {

    while (begin != end) {
        if (predicate(*stencil)) {
            *result = *begin;
            ++result;
        }

        ++begin;
        ++stencil;
    }
}

template <
    typename InIter, typename OutIter,
    std::predicate<typename InIter::value_type> PredicateT>
inline void
gather_if(InIter begin, InIter end, OutIter result, PredicateT predicate) {
    // treat input iterator as stencil
    gather_if(begin, end, result, begin, predicate);
}

template <
    typename InIter, typename OutIter,
    std::predicate<typename InIter::value_type> PredicateT>
inline void
gather_indices_if(
    InIter begin, InIter end, PredicateT predicate, OutIter result) {

    using IndexT = typename OutIter::container_type::value_type;

    const IndexT                           n = std::distance(begin, end);
    std::ranges::iota_view<IndexT, IndexT> indices{0, n};
    gather_if(indices.begin(), indices.end(), result, begin, predicate);
}

template <typename MaskIter, typename PRNG, typename Idx = long>
inline void
generate_shuffle_with_mask_indices(
    std::vector<Idx>& srcIdxs, std::vector<Idx>& targetIdxs,
    MaskIter maskBegin, MaskIter maskEnd, PRNG& prng) {

    // clear content, does not shrink allocated memory
    srcIdxs.clear();
    targetIdxs.clear();

    const auto is_true = [](const auto m) -> bool { return m; };

    gather_indices_if(maskBegin, maskEnd, is_true, std::back_inserter(srcIdxs));

    // copy indices to preserve them as source indices
    std::copy(srcIdxs.cbegin(), srcIdxs.cend(), std::back_inserter(targetIdxs));
    std::shuffle(targetIdxs.begin(), targetIdxs.end(), prng);
}

/// reorders according to arbitrary mapping of positions
template <typename InIter, typename IdxIter>
inline void
reorder(InIter begin, InIter end, IdxIter srcIdxsBegin, IdxIter srcIdxsEnd, IdxIter dstIdxsBegin) {
    const auto nIndices = std::distance(srcIdxsBegin, srcIdxsEnd);

    using ItemT = InIter::value_type;
    std::vector<ItemT> readOnlySrc{begin, end};

    for (int i{0}; i < nIndices; ++i) {
        const auto src = begin + srcIdxsBegin[i];
        const auto dst = readOnlySrc.begin() + dstIdxsBegin[i];

        *src = *dst;
    }
}

/// shuffles by predetermined order
template <typename InIter, typename IdxIter>
inline void
reorder(InIter begin, InIter end, IdxIter idx) {
    using ItemT = InIter::value_type;
    std::vector<ItemT> readOnlySrc{begin, end};

    const auto n = std::distance(begin, end);

    for (long i{0}; i < n; ++i)
        begin[i] = readOnlySrc[idx[i]];
}

/// indices:
/// preallocated helper vector
/// indices has element type of char, because vector<bool> is an exception
/// in c++ :(
template <typename InIter, typename MaskIter, typename PRNG>
inline void
shuffle_masked(
    InIter begin, InIter end, MaskIter maskBegin, PRNG& prng) {

    using IndexVec = std::vector<long>;

    // helper vectors
    // no preallocation since it is not certain, how many indices will be necessary
    // allocation will happen on-demand
    IndexVec srcIdxs, targetIdxs;

    const auto length  = std::distance(begin, end);
    const auto maskEnd = maskBegin + length;

    generate_shuffle_with_mask_indices(srcIdxs, targetIdxs, maskBegin, maskEnd, prng);

    reorder(begin, end, srcIdxs.cbegin(), srcIdxs.cend(), targetIdxs.cbegin());
}

/// mask - preallocated helper vector, must have the same size as arr
template <typename InIter, typename PRNG, typename Idx = long>
inline void
choice_shuffle(
    const float probability, InIter begin, InIter end, PRNG& prng, std::vector<char>& mask) {

    std::bernoulli_distribution dist(probability);

    // fill with random numbers
    std::generate(mask.begin(), mask.end(), [&] { return dist(prng); });
    shuffle_masked(begin, end, mask.cbegin(), prng);
}

/// srcIdxs, targetIdx - helper vectors
/// no preallocation since it is not certain, how many indices will be necessary
/// allocation will happen on-demand
template <typename MaskIter, typename PRNG, typename Idx = long>
inline void
rnd_shuffle_indices(
    std::vector<Idx>& srcIdxs, std::vector<Idx>& targetIdxs,
    const float probability, MaskIter maskBegin, MaskIter maskEnd, PRNG& prng) {
    std::bernoulli_distribution dist(probability);

    // fill with random numbers
    std::generate(maskBegin, maskEnd, [&] { return dist(prng); });
    generate_shuffle_with_mask_indices(srcIdxs, targetIdxs, maskBegin, maskEnd, prng);
}

template <typename Iter, typename IterMask>
inline void
swap_masked(Iter begin1, Iter end1, Iter begin2, IterMask mask) {
    while (begin1 != end1) {
        if (*mask)
            std::swap(begin1, begin2);

        ++begin1;
        ++begin2;
        ++mask;
    }
}
