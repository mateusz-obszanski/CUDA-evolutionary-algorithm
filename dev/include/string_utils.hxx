#pragma once
#include <algorithm>
#include <string>
#include <vector>

inline std::string
lpad(const std::size_t width, const std::string& str, const char padChar = ' ') {
    return std::string(width < str.length() ? 0 : width - str.length(), padChar) + str;
}

template <typename T>
concept TriviallyStringifiable = requires(const T& x) { std::to_string(x); };

template <TriviallyStringifiable T>
inline std::string
to_lpad(const std::size_t width, const T& x, const char padChar = ' ') {
    return lpad(width, std::to_string(x), padChar);
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
cmpStrByLength(const std::string& s1, const std::string& s2) noexcept {
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
                   [](const auto& x) { return std::to_string(x); });

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
