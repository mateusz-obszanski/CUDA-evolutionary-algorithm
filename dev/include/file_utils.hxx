#pragma once
#include "./common_concepts.hxx"
#include "./utils.hxx"
#include <algorithm>
#include <bit>
#include <fstream>
#include <iterator>
#include <vector>

struct AutoCloseFile {
    [[nodiscard]] AutoCloseFile(std::ofstream& stream) : stream(stream) {}
    ~AutoCloseFile() noexcept { stream.close(); }

private:
    std::ofstream& stream;
};

template <std::forward_iterator Iter>
    requires(not std::random_access_iterator<Iter>)
inline auto
save_data(std::ofstream& out, Iter begin, Iter end) {
    std::for_each(begin, end, [&](auto const& x) {
        out.write(std::bit_cast<const char*>(&x), sizeof(decltype(x)));
    });
    return out.bad();
}

template <std::random_access_iterator Iter>
inline auto
save_data(std::ofstream& out, Iter begin, Iter end) {
    using T = typename std::iterator_traits<Iter>::value_type;

    out.write(std::bit_cast<const char*>(iter_to_ptr_cast<T>(begin)),
              sizeof(T) * std::distance(begin, end));
    return out.bad();
}

template <PrimitiveType T>
inline auto
save_data(std::ofstream& out, std::vector<T> const& data) {
    out.write(std::bit_cast<const char*>(data.data()),
              static_cast<long>(sizeof(T) * data.size()));
    return out.bad();
}

template <std::forward_iterator Iter>
    requires(not std::random_access_iterator<Iter>)
inline auto
save_data(std::string& filename, Iter begin, Iter end) {
    std::ofstream out(filename, std::ofstream::binary);
    const auto    cleanup = AutoCloseFile(out);
    return save_data(out, begin, end);
}

template <typename T, typename Alloc>
inline auto
save_data(std::string& filename, std::vector<T, Alloc> const& data) {
    std::ofstream out(filename, std::ofstream::binary);
    const auto    cleanup = AutoCloseFile(out);
    return save_data(out, data);
}
