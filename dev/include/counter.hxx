#pragma once
#include <iterator>

template <typename Idx = int>
class Counter;

template <typename Idx>
class SimpleCounterIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = Idx;
    using pointer           = const Idx*;
    using reference         = const Idx&;

    SimpleCounterIterator() = delete;
    [[nodiscard]] constexpr SimpleCounterIterator(const Idx n, const Idx step = Idx(1)) noexcept
    : cnt{}, idx{n * (step > 0)}, step{step} {}

    [[nodiscard]] constexpr reference
    operator*() const noexcept { return idx; }

    // ++i
    constexpr const SimpleCounterIterator&
    operator++() noexcept {
        ++cnt;
        idx += step;
        return *this;
    }

    // i++
    constexpr SimpleCounterIterator
    operator++(int) noexcept {
        const auto tmp = *this;
        ++(*this);
        return tmp;
    }

    // --i
    constexpr const SimpleCounterIterator&
    operator--() noexcept {
        --cnt;
        idx += step;
        return *this;
    }

    // i--
    constexpr SimpleCounterIterator
    operator--(int) noexcept {
        const auto tmp = *this;
        ++(*this);
        return tmp;
    }

    [[nodiscard]] constexpr SimpleCounterIterator
    operator+(const Idx i) const noexcept {
        return {cnt + i, idx + step * i, step};
    }

    [[nodiscard]] constexpr SimpleCounterIterator
    operator-(const Idx i) const noexcept {
        return {cnt - i, idx - step * i, step};
    }

    [[nodiscard]] friend constexpr auto
    operator!=(const SimpleCounterIterator& a, const SimpleCounterIterator& b) noexcept {
        return a.cnt != b.cnt;
    }

    [[nodiscard]] friend constexpr auto
    operator==(const SimpleCounterIterator& a, const SimpleCounterIterator& b) noexcept {
        return a.cnt == b.cnt;
    }

    [[nodiscard]] friend constexpr auto
    operator<(const SimpleCounterIterator& a, const SimpleCounterIterator& b) noexcept {
        return a.cnt < b.cnt;
    }

    [[nodiscard]] friend constexpr auto
    operator>(const SimpleCounterIterator& a, const SimpleCounterIterator& b) noexcept {
        return a.cnt > b.cnt;
    }

    [[nodiscard]] friend constexpr auto
    operator<=(const SimpleCounterIterator& a, const SimpleCounterIterator& b) noexcept {
        return a.cnt <= b.cnt;
    }

    [[nodiscard]] friend constexpr auto
    operator>=(const SimpleCounterIterator& a, const SimpleCounterIterator& b) noexcept {
        return a.cnt >= b.cnt;
    }

private:
    Idx       cnt;
    Idx       idx;
    const Idx step;

    [[nodiscard]] constexpr SimpleCounterIterator(const Idx cnt, const Idx idx, const Idx step) noexcept
    : cnt{cnt}, idx{idx}, step{step} {}

    [[nodiscard]] constexpr SimpleCounterIterator(const SimpleCounterIterator& i) : SimpleCounterIterator(i.cnt, i.idx, i.step) {}

    friend constexpr SimpleCounterIterator
    Counter<Idx>::end() const noexcept;
};

template <typename Idx>
class Counter {
private:
    const Idx offset;
    const Idx n;
    const Idx step;

public:
    using index_type = Idx;
    using iterator   = SimpleCounterIterator<Idx>;

    Counter() = delete;

    [[nodiscard]] constexpr Counter(const Idx offset, const Idx n, const Idx step)
    // stop == n if incr else stop == 0 and step == -1 (countdown mode)
    : offset(offset), n(n * (step > 0)), step(step) {}

    [[nodiscard]] constexpr Counter(const Idx offset, const Idx n)
    : Counter(offset, n, 1) {}

    [[nodiscard]] constexpr Counter(const Idx n)
    : Counter(0, n, 1) {}

    [[nodiscard]] constexpr iterator
    begin() const noexcept { return {offset, step}; }

    [[nodiscard]] constexpr iterator
    end() const noexcept { return begin() + n - offset; }
};
