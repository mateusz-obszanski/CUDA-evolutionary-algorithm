#pragma once
#include "../matrix_utils.hxx"
#include "./ea_utils.hxx"
#include <algorithm>
#include <limits>
#include <type_traits>

/// Open TSP loss function - does not add cost from the last destination to the
/// 0 index
template <typename SolutionIter>
[[nodiscard]] inline float
calcLossOpen(const CostMx& costMx, const std::size_t nLocations,
             SolutionIter begin, SolutionIter end) {
    float loss = 0.0f;

    using Location = std::remove_const<
        typename std::iterator_traits<SolutionIter>::value_type>::type;

    // assuming that the starting point is 0
    // Location current = 0;
    int current = 0;

    const MatrixView mxView(costMx.begin(), nLocations);

    std::for_each(begin, end, [=, &current, &mxView, &loss](Location dst) {
        loss += mxView.get(current, dst);
        current = dst;
    });

    return loss;
}

template <typename SolutionIter>
[[nodiscard]] inline float
calcLossClosed(const CostMx& costMx, const std::size_t nLocations,
               SolutionIter begin, SolutionIter end) {
    const MatrixView mxView(costMx.begin(), nLocations);

    const auto last = *(end - 1);

    return calcLossOpen(costMx, nLocations, begin, end) + mxView.get(last, 0);
}

template <std::forward_iterator Individual>
inline void
calcPopulationLosses(std::vector<float>&            losses,
                     const std::vector<Individual>& population,
                     const CostMx& costMx, const int nLocations) {

    const auto nGenes = nLocations - 1;
    std::transform(
        population.cbegin(), population.cend(), losses.begin(), [&](auto ind) {
            return calcLossClosed(costMx, nLocations, ind, ind + nGenes);
        });
}

template <typename Individual>
inline std::vector<float>
calcPopulationLosses(const std::vector<Individual>& population,
                     const CostMx& costMx, const int nLocations) {

    std::vector<float> losses(population.size());
    calcPopulationLosses(losses, population, costMx, nLocations);

    return losses;
}

struct LossFnClosed {
public:
    const int           nLocations;
    const int           populationSize;
    const CostMx* const costMx;

    LossFnClosed() = delete;

    [[nodiscard]] LossFnClosed(const int nLocations, CostMx const* costMxPtr,
                               const int populationSize)
    : nLocations(nLocations),
      populationSize(populationSize),
      costMx(costMxPtr) {}

    [[nodiscard]] LossFnClosed(LossFnClosed const& other)
    : nLocations(other.nLocations),
      populationSize(other.populationSize),
      costMx(other.costMx) {}

    [[nodiscard]] LossFnClosed(LossFnClosed&& other)
    : nLocations(std::move(other.nLocations)),
      populationSize(std::move(other.populationSize)),
      costMx(std::move(other.costMx)) {}

    template <typename IndividualPtr>
    void
    operator()(std::vector<IndividualPtr> const& population,
               std::vector<float>&               values) const {

        calcPopulationLosses(values, population, *costMx, nLocations);
    }

    template <typename IndividualPtr>
    [[nodiscard]] std::vector<float>
    operator()(std::vector<IndividualPtr> const& population) const {
        std::vector<float> values(populationSize);
        (*this)(population, values);
        return values;
    }
};
