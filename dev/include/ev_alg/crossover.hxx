#pragma once
#include "../indexing_adapter.hxx"
#include "../rnd_utils.hxx"
#include "./ea_utils.hxx"
#include <iterator>
#include <optional>
#include <span>
#include <vector>

template <typename Iter, typename IterMask, typename PRNG>
inline void
rndMaskCrossover(Iter begin1, Iter end1, Iter begin2, IterMask mask,
                 const float prob, PRNG& prng) {
    const auto                  n = std::distance(begin1, end1);
    std::bernoulli_distribution dist(prob);
    // fill random mask
    // instead of generating mask array, mask value could be calculated as a
    // local variable
    std::generate(mask, mask + n, [&] { return dist(prng); });
    swap_masked(begin1, end1, begin2, mask);
}

template <typename Iter, typename PRNG>
inline void
rndMaskCrossover2(Iter begin1, Iter end1, Iter begin2, const float prob,
                  PRNG& prng) {
    const auto                             n = std::distance(begin1, end1);
    std::bernoulli_distribution            dist(prob);
    const std::ranges::iota_view<int, int> idx(0, n);

    const auto rndSwap = [&dist, &prng, &begin1, &begin2](const auto idx) {
        const auto shouldSwap = dist(prng);
        swap_if(shouldSwap, begin1[idx], *begin2[idx]);
    };

    // for parallel execution, pass prng by copy and offet by idx
    std::for_each(idx.begin(), idx.end(), rndSwap);
}

class Crossover {
public:
    [[nodiscard]] Crossover(float prob) : prob(prob) {}
    [[nodiscard]] Crossover(Crossover const&) = default;
    [[nodiscard]] Crossover(Crossover&&)      = default;

    template <typename Iter, typename PRNG>
    void
    operator()(Iter begin1, Iter end1, Iter begin2, PRNG& prng) {
        rndMaskCrossover2(begin1, end1, begin2, prob, prng);
    }

private:
    const float prob;
};

/**
 * 2-point PMX ordered crossover
 * based on: https://blog.x5ff.xyz/blog/ai-rust-javascript-pmx/ by Claus,
 * date of access: 01 June 2023
 *
 * Requires sequence minimum to be >= 1
 */
struct CrossoverPMX2Point {
    using GeneT          = int;
    using Parent         = std::span<const GeneT>;
    using WritableParent = std::span<GeneT>;
    using Child          = std::vector<GeneT>;
    /// Thanks to an index-based genome we can use a vector here.
    /// Use a std::unordered_map when the genome is a string or object
    using Mapping    = std::vector<std::optional<GeneT>>;
    using CrossPoint = int;

    /// parents must have equal size
    template <typename PRNG>
    [[nodiscard]] std::pair<Child, Child>
    operator()(Parent p1, Parent p2, PRNG& prng) const {
        return cross<PRNG>(p1, p2, prng);
    }

    template <typename PRNG>
    [[nodiscard]] inline static std::pair<Child, Child>
    cross(Parent p1, Parent p2, PRNG& prng) {
        const auto n = p1.size();

        // The crossover points
        //  x1 |     | x2 (excl)
        // [1, 2, 3, 4, 5]
        const auto&& [x1, x2] = randspan<CrossPoint>(n, prng);

        return {breed(p1, p2, x1, x2), breed(p2, p1, x1, x2)};
    }

    template <typename PRNG>
    inline static void
    cross_inplace(WritableParent p1, WritableParent p2, PRNG& prng) {
        const auto&& [c1, c2] = cross(p1, p2, prng);
        std::copy(c1.cbegin(), c1.cend(), p1.begin());
        std::copy(c2.cbegin(), c2.cend(), p2.begin());
    }

private:
    [[nodiscard]] inline static Child
    breed(Parent p1, Parent p2, const CrossPoint x1, const CrossPoint x2) {
        const auto n = p1.size();

        Child   offspring(n); // filled with 0s
        Mapping mapping(n);   // filled with std::nullopt

        // - 1, because with current encoding, numbers start from 1
        const auto gene_to_index = [](auto gene) { return gene - 1; };

        RefIndexingAdapter mappingAdapter(mapping, gene_to_index);

        // Inheritance! This sets the values within the crossover zone.
        for (int i{x1}; i < x2; ++i) {
            offspring[i] = p2[i];

            // Put the values that "should have been there" into the map.
            mappingAdapter[p2[i]] = p1[i];
        }

        // left of the crossover zone
        scan_for_duplicates(0, x1, p1, offspring, mappingAdapter);
        // right -||-
        scan_for_duplicates(x2, n, p1, offspring, mappingAdapter);

        return offspring;
    }

    inline static void
    scan_for_duplicates(const int begin, const int end, Parent parent1,
                        Child& offspring, auto& mapping) {

        for (int i{begin}; i < end; ++i) {
            auto                  overwritten = mapping[parent1[i]];
            decltype(overwritten) lastOverwritten;

            // look for non-duplicated value
            while (overwritten.has_value()) {
                lastOverwritten = overwritten;
                overwritten     = mapping[overwritten.value()];
            }

            // if was not duplicated, write from parent, else write found value
            // lastOverwritten - last overwritten gene that does not overwrite
            // other genes
            offspring[i] = lastOverwritten.value_or(parent1[i]);
        }
    }
};

struct CrossoverPMX2PointAdapter {
    template <typename Iter, typename PRNG>
    void
    operator()(Iter begin1, Iter end1, Iter begin2, PRNG& prng) const {
        const std::size_t n = std::distance(begin1, end1);

        CrossoverPMX2Point::cross_inplace<PRNG>({begin1, n}, {begin2, n}, prng);
    }
};
