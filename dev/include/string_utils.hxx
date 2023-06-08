#pragma once
#include "./common_concepts.hxx"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

template <std::integral T>
[[nodiscard]] inline std::string
to_hex(T x) {
    return (std::ostringstream() << "0x" << std::hex << x).str();
}

template <TriviallyStringifiable T>
[[nodiscard]] inline std::string
stringify(T const& x) {
    return std::to_string(x);
}

template <SStreamStringifiable T>
[[nodiscard]] inline std::string
stringify(T const& x) {
    return (std::ostringstream() << x).str();
}

template <typename T>
[[nodiscard]] inline std::string
stringify(T const* const ptr) {
    return to_hex(std::bit_cast<std::uintptr_t>(ptr));
}

inline std::string
lpad(const std::size_t width, const std::string& str,
     const char padChar = ' ') {
    return std::string(width < str.length() ? 0 : width - str.length(),
                       padChar) +
           str;
}

template <typename T>
inline std::string
to_lpad(const std::size_t width, const T& x, const char padChar = ' ') {
    return lpad(width, stringify(x), padChar);
}

inline std::string
repeat_char(const std::size_t n, const char c) {
    return std::string(n, c);
}

inline std::string
spaces(const std::size_t n) {
    return repeat_char(n, ' ');
}

inline bool
cmp_str_by_length(const std::string& s1, const std::string& s2) noexcept {
    return s1.length() < s2.length();
}

template <typename Iter>
inline std::vector<std::string>
stringify_many(Iter begin, Iter end) {
    std::vector<std::string> strings;
    const auto               nElems = std::distance(begin, end);
    strings.reserve(nElems);

    // fill with stringified elements
    std::transform(begin, end, std::back_inserter(strings),
                   [](const auto& x) { return stringify(x); });

    return strings;
}

template <std::convertible_to<bool> T>
[[nodiscard]] inline constexpr std::string
pretty_bool(const T x) {
    return x ? "true" : "false";
}

template <std::convertible_to<bool> T>
[[nodiscard]] inline constexpr std::string
yes_no(const T x) {
    return x ? "yes" : "no";
}

[[nodiscard]] inline constexpr std::string
str_or(std::string const& str, std::string const& alternative) noexcept {
    return str.size() ? str : alternative;
}

[[nodiscard]] inline constexpr std::string_view
str_or(std::string_view str, std::string_view alternative) noexcept {
    return str.size() ? str : alternative;
}

// trim from start (in place)
inline void
ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
}

// trim from end (in place)
inline void
rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            s.end());
}

// trim from both ends (in place)
inline void
trim(std::string& s) {
    rtrim(s);
    ltrim(s);
}

// trim from start (copying)
inline std::string
ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
inline std::string
rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
inline std::string
trim_copy(std::string s) {
    trim(s);
    return s;
}
