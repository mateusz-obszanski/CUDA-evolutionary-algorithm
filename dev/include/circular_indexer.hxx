#pragma once

inline long
resolveCircularIdx(const long idx, const long collectionSize) {
    // branchless version, equivalent to the following:
    // if (idx < 0)
    //     return collectionSize - (-idx) % collectionSize;
    // else
    //     return idx % collectionSize;
    const auto isNegative = idx < 0;
    const auto sign =
        (1 - (isNegative << 1)); // 1 if index > 0, -1 if index < 0, 0 otherwise
    return collectionSize * isNegative + sign * ((sign * idx) % collectionSize);
}

class CircularIndexer {
public:
    const long collectionSize;

    CircularIndexer() = delete;
    [[nodiscard]] CircularIndexer(const long collectionSize) noexcept
    : collectionSize{collectionSize} {};

    long
    operator()(const long idx) const {
        return resolveCircularIdx(idx, collectionSize);
    }
};
