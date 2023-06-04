#pragma once
#include "./iter_utils.hxx"
#include <string>
#include <vector>

/// Permutation can be encoded as an inversion
/// vector of the same length or shorter by one,
/// because the last element of inversion vector
/// is always 0, which can be implicitly assumed
enum class InversionVectorCoding { SAME, SHORT };

[[nodiscard]] inline constexpr std::string
inversion_vec_coding_to_str(InversionVectorCoding coding) noexcept {
    return coding == InversionVectorCoding::SHORT ? "SHORT" : "SAME";
}

[[nodiscard]] inline constexpr std::string
inversion_vec_coding_to_full_str(InversionVectorCoding coding) noexcept {
    return "InversionVectorCoding::" + inversion_vec_coding_to_str(coding);
}

/// inv - output
/// inv must point to memory with space for n (SAME) or n-1 (SHORT) elemenst
template <typename Iter, typename IterOut,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
inline void
permutation_to_inversion_vector(Iter perm, int n, IterOut inv) {
    if constexpr (CODING == InversionVectorCoding::SHORT)
        --n;

    for (int i{0}; i < n; ++i) {
        int m = 0;
        // temporary local variables to avoid slow memory access
        auto perm_m = perm[m];
        auto inv_i  = inv[i];

        // for all perm elements on the left of i
        while (perm_m != i) {
            // branchless increment if condition is true
            inv_i += perm_m > i;
            perm_m = perm[++m];
        }

        inv[i] = inv_i;
    }
}
///
/// inv - output
/// inv must point to memory with space for n (SAME) or n-1 (SHORT) elemenst
template <typename T, typename IterOut,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
inline void
permutation_to_inversion_vector(std::span<T const> perm, IterOut inv) {
    permutation_to_inversion_vector(perm.data(), perm.size(), inv);
}

/// assumption: perm elements in {0, 1, ..., n - 1}
template <typename Iter,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
[[nodiscard]] inline std::vector<int>
permutation_to_inversion_vector(Iter perm, int n) {
    int inv_length;

    if constexpr (CODING == InversionVectorCoding::SHORT)
        inv_length = n - 1;
    else
        inv_length = n;

    std::vector<int> inv(inv_length);

    permutation_to_inversion_vector<Iter, decltype(inv.begin()), CODING>(
        perm, n, inv.begin());

    return inv;
}

template <typename T,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
[[nodiscard]] inline auto
permutation_to_inversion_vector(std::span<T const> perm) {
    return permutation_to_inversion_vector<decltype(perm.data()), CODING>(
        perm.data(), perm.size());
}

/// output: perm
template <typename Iter, typename IterOut,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
inline void
inversion_vector_to_permutation(Iter inv, int n, IterOut perm) {
    std::size_t perm_size;

    if constexpr (CODING == InversionVectorCoding::SHORT)
        perm_size = n + 1;
    else
        perm_size = n;

    auto freeIndices = range_vec<int>(perm_size);

    for (int i{0}; i < n; ++i) {
        const auto inv_i         = inv[i];
        perm[freeIndices[inv_i]] = i;
        freeIndices.erase(freeIndices.begin() + inv_i);
    }

    if constexpr (CODING == InversionVectorCoding::SHORT)
        perm[freeIndices[0]] = n;
}
///
/// output: perm
template <typename T, typename IterOut,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
inline void
inversion_vector_to_permutation(std::span<T const> inv, IterOut perm) {
    inversion_vector_to_permutation<decltype(inv.data()), IterOut, CODING>(
        inv.data(), inv.size(), perm);
}

template <typename Iter,
          InversionVectorCoding CODING = InversionVectorCoding::SHORT>
[[nodiscard]] inline std::vector<int>
inversion_vector_to_permutation(Iter inv, int n) {
    std::size_t perm_size;

    if constexpr (CODING == InversionVectorCoding::SHORT)
        perm_size = n + 1;
    else
        perm_size = n;

    std::vector<int> permutation(perm_size);

    inversion_vector_to_permutation<Iter, decltype(permutation.begin()),
                                    CODING>(inv, n, permutation.begin());

    return permutation;
}

template <InversionVectorCoding CODING = InversionVectorCoding::SHORT>
[[nodiscard]] inline auto
inversion_vector_to_permutation(std::span<int const> const inv) {
    return inversion_vector_to_permutation<decltype(inv.data()), CODING>(
        inv.data(), inv.size());
}

template <InversionVectorCoding CODING = InversionVectorCoding::SHORT>
struct PermutationCoder {
    template <typename Iter>
    [[nodiscard]] inline static auto
    encode(Iter perm, int n) {
        return permutation_to_inversion_vector<Iter, CODING>(perm, n);
    }

    [[nodiscard]] inline static auto
    encode(std::span<const int> const perm) {
        return permutation_to_inversion_vector<decltype(perm)::value_type,
                                               CODING>(perm);
    }

    template <typename Iter>
    [[nodiscard]] inline static auto
    decode(Iter inv, int n) {
        return inversion_vector_to_permutation<Iter, CODING>(inv, n);
    }

    [[nodiscard]] inline static auto
    decode(std::span<int const> const inv) {
        return inversion_vector_to_permutation<CODING>(inv);
    }

    [[nodiscard]] inline static auto
    get_coding() noexcept {
        return CODING;
    }

    [[nodiscard]] inline static auto
    get_coding_str() noexcept {
        return inversion_vec_coding_to_str(get_coding());
    }

    [[nodiscard]] inline static auto
    get_coding_full_str() noexcept {
        return inversion_vec_coding_to_full_str(get_coding());
    }
};

using PermutationCoderSame  = PermutationCoder<InversionVectorCoding::SAME>;
using PermutationCoderShort = PermutationCoder<InversionVectorCoding::SHORT>;

/// Abstract version
struct AbstractPermutationCoder {
    [[nodiscard]] virtual std::vector<int>
    encode(std::span<int const> perm) const = 0;

    [[nodiscard]] virtual std::vector<int>
    decode(std::span<int const> inv) const = 0;

    [[nodiscard]] virtual InversionVectorCoding
    get_coding() const noexcept = 0;

    [[nodiscard]] virtual std::string
    get_coding_str() const noexcept = 0;

    [[nodiscard]] virtual std::string
    get_coding_full_str() const noexcept = 0;

    virtual ~AbstractPermutationCoder() = default;
};

template <InversionVectorCoding CODING = InversionVectorCoding::SHORT>
struct PolymorphicPermutationCoder : public AbstractPermutationCoder {
    [[nodiscard]] virtual std::vector<int>
    encode(std::span<int const> const perm) const override {
        return mCoder.encode(perm);
    }

    [[nodiscard]] virtual std::vector<int>
    decode(std::span<int const> const inv) const override {
        return mCoder.decode(inv);
    }

    [[nodiscard]] virtual InversionVectorCoding
    get_coding() const noexcept override {
        return mCoder.get_coding();
    };

    [[nodiscard]] virtual std::string
    get_coding_str() const noexcept override {
        return mCoder.get_coding_str();
    };

    [[nodiscard]] virtual std::string
    get_coding_full_str() const noexcept override {
        return mCoder.get_coding_full_str();
    };

private:
    [[no_unique_address]] const PermutationCoder<CODING> mCoder;
};

using PolymorphicPermutationCoderSame =
    PolymorphicPermutationCoder<InversionVectorCoding::SAME>;
using PolymorphicPermutationCoderShort =
    PolymorphicPermutationCoder<InversionVectorCoding::SHORT>;

/// inv[i] must be < n - 1 - i (only n - 1 - i elements on left can be
/// greater than i in permutation)
template <typename Iter>
inline void
repair_inversion_vector(Iter begin, Iter end) {
    using T = typename std::iterator_traits<Iter>::value_type;

    const T                            n = std::distance(begin, end);
    const std::ranges::iota_view<T, T> idx(0, n);

    const auto moduloIdx = [=, &begin](const auto& i) {
        return begin[i] % n - 1 - i;
    };

    std::transform(idx.begin(), idx.end(), begin, moduloIdx);
}
