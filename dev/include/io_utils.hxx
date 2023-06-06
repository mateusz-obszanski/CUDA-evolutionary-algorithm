#pragma once
#include "./string_utils.hxx"
#include <iostream>

template <typename T>
inline auto&
print_padded(const std::size_t nPad, const T& x,
             std::ostream& out = std::cout) {
    return out << to_lpad(nPad, x);
}

inline auto&
print_padding(const std::size_t n, const char padChar = ' ',
              std::ostream& stream = std::cout) {
    return stream << std::string(n, padChar);
}

inline auto&
print_separation_line(const std::size_t n, const char sep = '-',
                      std::ostream& out = std::cout) {
    return out << '\n' << repeat_char(n, sep) << '\n';
}
