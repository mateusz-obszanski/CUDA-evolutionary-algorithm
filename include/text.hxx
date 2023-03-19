#pragma once

#include <experimental/source_location> // this has been merged to c++20, but does not work with nvcc
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

namespace text {

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
         << ':' << lineInfo.column() << ")`" << lineInfo.function_name() << '`';

    return buff.str();
}

} // namespace text
