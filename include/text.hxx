#pragma once

#include "./types/concepts.hxx"
#include <experimental/source_location> // this has been merged to c++20, but does not work with nvcc
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

namespace text {

template <types::concepts::Stringifiable T, typename Alloc = std::pmr::polymorphic_allocator<T>>
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

/// @brief For not stringifiable types - placeholders
/// @tparam Alloc
/// @tparam T
/// @param vec
/// @param end
/// @param out
template <types::concepts::NotStringifiable T, typename Alloc = std::pmr::polymorphic_allocator<T>>
inline void
printVec(
    const std::vector<T, Alloc>& vec,
    const std::string&           end         = "\n",
    std::ostream&                out         = std::cout,
    const std::string&           placeholder = "<?>") {

    out << '[';

    const auto vecLen = vec.size();

    for (size_t i{0}; i < (vecLen == 0 ? 0 : vecLen - 1); ++i)
        out << placeholder << ", ";

    if (vecLen != 0)
        out << placeholder;

    out << ']' << end;
}

template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
inline std::string
fmtVec(const std::vector<T, Alloc>& vec) {
    std::stringstream buff;

    printVec(vec, "", buff);

    return buff.str();
}

namespace {

using std::experimental::source_location;

}

[[nodiscard]] std::string
fmtLineInfo(const source_location& lineInfo) noexcept {
    std::stringstream buff;

    buff << "file " << lineInfo.file_name() << '(' << lineInfo.line()
         << ':' << lineInfo.column() << ") `" << lineInfo.function_name() << '`';

    return buff.str();
}

} // namespace text
