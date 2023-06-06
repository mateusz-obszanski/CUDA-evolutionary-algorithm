#pragma once

template <typename Indexable, typename IndexTransform>
struct RefIndexingAdapter {
    RefIndexingAdapter(Indexable& wrapped, IndexTransform transform)
    : wrapped(wrapped), transform(transform) {}

    auto&
    operator[](auto idx) {
        return wrapped[transform(idx)];
    }

    auto const&
    operator[](auto idx) const {
        return wrapped[transform(idx)];
    }

private:
    Indexable&     wrapped;
    IndexTransform transform;
};
