#pragma once

#include "../concepts.hxx"
#include "./shape.hxx"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory_resource>
#include <ostream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

namespace utils {
    template <concepts::Stringifiable T, concepts::InputIter<T> InIter>
    std::string
    join(InIter begin, InIter end, const std::string& delim = ", ") {
        std::stringstream buff;
        std::copy(begin, end, std::ostream_iterator<T>(buff, delim.c_str()));

        return buff.str();
    }

    template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
    inline void
    printVec(
        const std::vector<T, Alloc>& vec,
        const std::string&           end = "\n",
        std::ostream&                out = std::cout) {

        out << '[';

        std::copy(
            vec.cbegin(), std::prev(vec.cend()),
            std::ostream_iterator<T>(out, ", "));

        if (vec.size() != 0)
            out << vec.back();

        out << ']' << end;
    }

    template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
    inline std::string
    fmtVec(const std::vector<T, Alloc>& vec) {
        std::stringstream buff;

        printVec<T>(vec, "", buff);

        return buff.str();
    }

    template <
        typename T,
        std::size_t NDims,
        typename SizeT = std::size_t,
        typename Alloc = std::pmr::polymorphic_allocator<T>>
    inline std::string
    fmtShapedVec(
        const Shape<NDims, SizeT>&   shape,
        const std::vector<T, Alloc>& v) {

        checkShapesCompatibility(shape, {v.size()});

        const auto [lenX, lenY, lenZ] = shape;

        const auto nRows = lenY * lenZ;

        std::string result{};

        // TODO

        // for (const auto z : std::ranges::views::iota(lenZ)) {

        //     for (const auto y : std::ranges::views::iota(lenY)) {
        //     }
        // }

        // for (const auto i : std::ranges::views::iota(nRows - 1)) {
        //     const auto beg = i*;
        //     const auto end = beg;
        // }
    }
} // namespace utils
