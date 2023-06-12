#pragma once

#include "./common_concepts.hxx"
#include <iostream>
#include <iterator>
#include <source_location> // this has been merged to c++20, but does not work with nvcc - in case of errors, prefix with "experimental/"
#include <sstream>
#include <vector>

namespace text {

template <IterValSStreamStringifiable Iter>
inline void
printIter(Iter begin, Iter end, const std::string& ending = "\n",
          std::ostream& out = std::cout) {

    out << '[';

    std::copy(begin, std::prev(end),
              std::ostream_iterator<typename Iter::value_type>(out, ", "));

    if (std::distance(begin, end) != 0)
        out << *end;

    out << ']' << ending;
}

/// @brief For not stringifiable types - placeholders
/// @tparam Alloc
/// @tparam T
/// @param vec
/// @param end
/// @param out
template <NotIterValSStreamStringifiable Iter>
inline void
printIter(Iter begin, Iter end, const std::string& ending = "\n",
          std::ostream&      out         = std::cout,
          const std::string& placeholder = "<?>") {

    out << '[';

    const auto length = std::distance(begin, end);

    for (size_t i{0};
         i < static_cast<decltype(i)>(length == 0 ? 0 : length - 1); ++i)
        out << placeholder << ", ";

    if (length != 0)
        out << placeholder;

    out << ']' << ending;
}

template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
inline void
printVec(const std::vector<T, Alloc>& vec, const std::string& end = "\n",
         std::ostream& out = std::cout) {

    printIter(vec.cbegin(), vec.cend(), end, out);
}

template <typename T, typename Alloc = std::pmr::polymorphic_allocator<T>>
inline std::string
fmtVec(const std::vector<T, Alloc>& vec) {
    std::stringstream buff;

    printVec<T, Alloc>(vec, "", buff);

    return buff.str();
}

namespace {

using source_location =
    std::source_location; // or std::experimental::source_location

}

[[nodiscard]] std::string
fmtLineInfo(const source_location& lineInfo) noexcept {
    std::stringstream buff;

    buff << "file " << lineInfo.file_name() << '(' << lineInfo.line() << ':'
         << lineInfo.column() << ") `" << lineInfo.function_name() << '`';

    return buff.str();
}

} // namespace text
