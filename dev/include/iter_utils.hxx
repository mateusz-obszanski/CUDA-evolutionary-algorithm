#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <ranges>
#include <vector>

template <typename Iter>
inline void
printIter(Iter begin, Iter end) {
    using value_t = typename Iter::value_type;
    std::cout << '[';
    std::copy(begin, end, std::ostream_iterator<value_t>(std::cout, ", "));
    std::cout << "]\n";
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
printContainer(const Container& c) {
    printIter(c.cbegin(), c.cend());
}
