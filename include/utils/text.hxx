#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory_resource>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace utils {
    template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
    inline void
    printVec(const std::vector<T, Alloc>& vec, std::ostream& out = std::cout) {
        out << '[';

        std::copy(
            vec.cbegin(), std::prev(vec.cend()), std::ostream_iterator<T>(out, ", "));

        if (vec.size() != 0)
            out << vec.back();

        out << ']';
    }

    template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
    inline std::string
    fmtVec(const std::vector<T, Alloc>& vec) {
        std::stringstream buff;

        printVec<T>(vec, buff);

        return buff.str();
    }
} // namespace utils
