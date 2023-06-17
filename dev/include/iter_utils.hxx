#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <ranges>
#include <vector>

template <typename Iter>
inline void
print_iter(Iter begin, Iter end, std::ostream& out = std::cout) {
    using T = typename std::iterator_traits<Iter>::value_type;

    out << '[';
    const auto nextToLast = std::prev(end);
    std::copy(begin, nextToLast, std::ostream_iterator<T>(out, ", "));

    if (std::distance(begin, end) != 0)
        out << *nextToLast;

    out << ']';
}

template <typename Iter>
inline void
println_iter(Iter begin, Iter end, std::ostream& out = std::cout) {
    print_iter(begin, end, out);
    out << '\n';
}

template <typename Container>
concept ConstIterable = requires(Container c) {
    c.cbegin();
    c.cend();
};

template <typename Container>
concept MutIterable = requires(Container c) {
    c.begin();
    c.end();
};

template <typename Container>
concept RandomAccessContainer = requires(Container c, int i) {
    { c[i] } -> std::same_as<typename Container::value_type>;
};

template <ConstIterable Container>
inline void
print_container(const Container& c, std::ostream& out = std::cout) {
    print_iter(c.cbegin(), c.cend(), out);
}

template <ConstIterable Container>
inline void
println_container(const Container& c, std::ostream& out = std::cout) {
    println_iter(c.cbegin(), c.cend(), out);
}

/// stop >= start
template <typename Idx = int>
[[nodiscard]] inline std::vector<Idx>
range_vec(const Idx start, const Idx stop) {
    std::vector<Idx> result;
    result.reserve(stop - start);
    const std::ranges::iota_view indices(start, stop);
    std::for_each(indices.begin(), indices.end(),
                  [&result](const auto& i) { result.push_back(i); });

    return result;
}

template <typename Idx = int>
[[nodiscard]] inline std::vector<Idx>
range_vec(const Idx stop) {
    return range_vec(0, stop);
}
