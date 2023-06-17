#pragma once
#include <algorithm>
#include <exception>
#include <limits>
#include <string>

constexpr auto INF = std::numeric_limits<float>::infinity();

struct StopCondition {
    [[nodiscard]] StopCondition(
        unsigned int maxIters,
        unsigned int maxItersWithoutImprovement =
            std::numeric_limits<unsigned int>::max()) noexcept
    : bestLoss(INF),
      iteration(0),
      iterationSinceImprovement(0),
      maxIters(maxIters),
      maxItersWithoutImprovement(maxItersWithoutImprovement),
      stopReason(_StopReason::DID_NOT_STOP) {}

    [[nodiscard]] bool
    operator()(float currentBestLoss) {
        const bool didNotImprove = bestLoss <= currentBestLoss;
        bestLoss                 = didNotImprove ? bestLoss : currentBestLoss;
        iterationSinceImprovement += didNotImprove;

        const bool noImprovement =
            iterationSinceImprovement >= maxItersWithoutImprovement;
        const bool reachedMaxIters = iteration >= maxIters;

        const bool shouldStop = noImprovement | reachedMaxIters;

        ++iteration;

        if (shouldStop) [[unlikely]] {
            if (noImprovement)
                stopReason = _StopReason::NO_IMPROVEMENT;
            else
                stopReason = _StopReason::REACHED_MAX_ITERS;
        }

        return shouldStop;
    }

    enum class StopReason { REACHED_MAX_ITERS, NO_IMPROVEMENT };

    struct DidNotReachStopConditionError : public std::exception {
        const char*
        what() const noexcept override {
            return "stop condition has not been reached yet";
        }
    };

    [[nodiscard]] StopReason
    get_stop_reason() const {
        if (stopReason == _StopReason::DID_NOT_STOP) [[unlikely]]
            throw DidNotReachStopConditionError();

        return static_cast<StopReason>(stopReason);
    }

    [[nodiscard]] inline static std::string
    stop_reason_to_str(StopReason reason) {
        const char* names[] = {"REACHED_MAX_ITERS", "NO_IMPROVEMENT"};
        const auto  nameIdx = static_cast<std::size_t>(reason);
        return names[nameIdx];
    }

    [[nodiscard]] std::string
    get_stop_reason_str() const {
        return stop_reason_to_str(get_stop_reason());
    }

    [[nodiscard]] auto
    get_iters() const noexcept {
        return iteration;
    }

    [[nodiscard]] auto
    get_iters_since_improvement() const noexcept {
        return iterationSinceImprovement;
    }

private:
    // adds the possibility that stop condition has not been reached yet
    enum class _StopReason { REACHED_MAX_ITERS, NO_IMPROVEMENT, DID_NOT_STOP };

    float              bestLoss;
    unsigned int       iteration;
    unsigned int       iterationSinceImprovement;
    const unsigned int maxIters;
    const unsigned int maxItersWithoutImprovement;
    _StopReason        stopReason;
};
