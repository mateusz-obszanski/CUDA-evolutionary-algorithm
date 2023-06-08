#pragma once
#include "../common_concepts.hxx"
#include "../utils.hxx"
#include <algorithm>
#include <fstream>
#include <limits>

struct StandardStats {
    const float min;
    const float max;
    const float mean;
    const float variance;

    template <IterValConvertibleTo<float> Iter>
    [[nodiscard]] inline static StandardStats
    from_iter(Iter begin, Iter end) {
        using Limits       = std::numeric_limits<float>;
        constexpr auto inf = Limits::infinity();

        float min         = inf;
        float max         = -inf;
        float sum         = 0;
        float sumOfSquare = 0;

        std::for_each(begin, end, [&](const float x) {
            min = std::min(x, min);
            max = std::max(x, max);
            sum += x;
            sumOfSquare += square<float>(x);
        });

        const auto  n    = static_cast<float>(std::distance(begin, end));
        const float mean = sum / n;

        const auto meanOfSquare = sumOfSquare / n;
        const auto squareOfMean = square(mean);

        const float variance = meanOfSquare - squareOfMean;

        return {min, max, mean, variance};
    }

    void
    write_binary(std::ofstream& file) const {
        file.write(std::bit_cast<char*>(&min), sizeof(decltype(min)));
        file.write(std::bit_cast<char*>(&max), sizeof(decltype(max)));
        file.write(std::bit_cast<char*>(&mean), sizeof(decltype(mean)));
        file.write(std::bit_cast<char*>(&variance), sizeof(decltype(variance)));
    }
};
