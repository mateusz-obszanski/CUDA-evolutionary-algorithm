#pragma once

#include <concepts>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace device {
namespace mask {

/// @brief Initializes iterable of indices with indices, where mask is nonzero.
/// @tparam MaskIter
/// @tparam IdxIter
/// @param maskBegin
/// @param n
/// @param idxBegin memory underneath has to have size equal `n`
/// @return new end iterator for memory underneath `idxBegin`
template <typename MaskIter, typename IdxIter>
    requires std::convertible_to<typename MaskIter::value_type, bool> and
             std::integral<typename IdxIter::value_type>
inline IdxIter
mask_indices_n(MaskIter maskBegin, const std::size_t n, IdxIter idxBegin) {
    using Idx   = typename IdxIter::value_type;
    using MaskT = typename MaskIter::value_type;

    thrust::counting_iterator<Idx> firstIdx{};

    return thrust::copy_if(
        firstIdx, firstIdx + n, maskBegin,
        idxBegin, [] __device__(const MaskT m) { return static_cast<bool>(m); });
}

} // namespace mask
} // namespace device