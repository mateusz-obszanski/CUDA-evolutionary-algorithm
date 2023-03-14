#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

namespace utils {
    template <std::size_t N, typename SizeT = std::size_t>
    using Shape = std::array<SizeT, N>;

    template <typename T, std::size_t N, typename SizeT = std::size_t>
    concept IsShape = std::same_as<T, Shape<N, SizeT>>;

    template <std::size_t N, typename SizeT = std::size_t>
    [[nodiscard]] inline std::string
    fmtShape(const Shape<N, SizeT>& shape) {
        std::string result = "[";

        for (std::size_t i{0}; i < N - 1; ++i)
            result += std::to_string(shape[i]) + " x ";

        // last one
        if (N != 0)
            result += shape[N - 1];

        return result + "]";
    }

    template <std::size_t N, typename SizeT = std::size_t>
    [[nodiscard]] constexpr inline SizeT
    shapeToSize(const Shape<N, SizeT>& s) {
        return std::reduce(s.cbegin(), s.cend(), 1, std::multiplies<SizeT>{});
    }

    template <std::size_t N, typename SizeT = std::size_t>
    class IncompatibleShapesError : public std::exception {
    private:
        using shape_t = utils::Shape<N, SizeT>;

        const shape_t s1;
        const shape_t s2;

        const std::string msg;

    public:
        [[nodiscard]] IncompatibleShapesError(const shape_t& s1, const shape_t& s2)
        : s1{s1},
          s2{s2},
          msg{"IncompatibleShapesError: " + utils::fmtShape(s1) + " and " + utils::fmtShape(s2)} {}

        const char*
        what() const noexcept override {
            return msg.c_str();
        }
    };

    template <std::size_t N1, std::size_t N2, typename SizeT1 = std::size_t, typename SizeT2 = SizeT1>
    void
    checkShapesCompatibility(const Shape<N1, SizeT1>& s1, const Shape<N2, SizeT2>& s2) {
        if constexpr (shapeToSize(s1) == shapeToSize(s2))
            return;

        throw IncompatibleShapesError(s1, s2);
    }
} // namespace utils
