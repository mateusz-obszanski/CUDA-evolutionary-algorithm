#pragma once
#include "./string_utils.hxx"
#include <iostream>

template <TriviallyStringifiable T>
inline auto&
printPadded(const std::size_t nPad, const T& x, std::ostream& out = std::cout) {
    return out << to_lpad(nPad, x);
}

inline auto&
printPadding(const std::size_t n, const char padChar = ' ', std::ostream& stream = std::cout) {
    return stream << std::string(n, padChar);
}

inline auto&
printSeparationLine(const std::size_t n, const char sep = '-', std::ostream& out = std::cout) {
    return out << '\n'
               << repeat_char(n, sep) << '\n';
}
