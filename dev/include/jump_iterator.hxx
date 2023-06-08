#pragma once
#include "./common_concepts.hxx"
#include <bits/iterator_concepts.h>
#include <concepts>
#include <iterator>

template <typename WrappedIter>
struct DispatchJumpIteratorTag {
    using tag = std::iterator_traits<WrappedIter>::iterator_category;
};

template <typename WrappedIter>
    requires std::contiguous_iterator<WrappedIter>
struct DispatchJumpIteratorTag<WrappedIter> {
    // downgrade iterator tag, because JumpIterator jumps over contiguous
    // memory, so it itself is not an contiguous iterator
    using tag = std::random_access_iterator_tag;
};

template <std::forward_iterator Iter>
struct JumpIterator {
private:
    using Traits = std::iterator_traits<Iter>;

public:
    using iterator_category = DispatchJumpIteratorTag<Iter>::tag;
    using difference_type   = Traits::difference_type;
    using value_type        = Traits::value_type;
    using pointer           = Iter;
    using reference         = Traits::reference;
    using StrideT           = long;

    JumpIterator(pointer ptr, StrideT jumpLength)
    : wrappedIter(ptr), stride(jumpLength) {}
    JumpIterator(JumpIterator const& other)
        requires(std::copyable<Iter>)
    : wrappedIter(other.wrappedIter), stride(other.stride) {}
    JumpIterator(JumpIterator&& other)
        requires(std::movable<Iter>)
    : wrappedIter(std::move(other.wrappedIter)), stride(other.stride) {}

    operator pointer() { return wrappedIter; }

    auto
    jump_length() const {
        return stride;
    }

    reference
    operator*() const {
        return *wrappedIter;
    }

    pointer
    operator->() {
        return wrappedIter;
    }

    JumpIterator&
    operator++()
        requires(std::forward_iterator<Iter>)
    {
        if constexpr (InplaceAddable<Iter, StrideT>)
            wrappedIter += stride;
        else
            std::advance(wrappedIter, stride);
        return *this;
    }

    JumpIterator
    operator++(int)
        requires(PreIncrementable<JumpIterator> and std::copyable<JumpIterator>)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    JumpIterator&
    operator--()
        requires(std::bidirectional_iterator<Iter>)
    {
        if constexpr (InplaceSubtractable<Iter, StrideT>)
            wrappedIter -= stride;
        else
            std::advance(wrappedIter, -stride);
        return *this;
    }

    JumpIterator
    operator--(int)
        requires(PreDecrementable<JumpIterator> and std::copyable<JumpIterator>)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    JumpIterator&
    operator+=(StrideT delta)
        requires(std::forward_iterator<Iter>)
    {
        if constexpr (InplaceAddable<Iter, StrideT>)
            wrappedIter += delta * stride;
        else
            std::advance(wrappedIter, delta * stride);
        return *this;
    }

    JumpIterator
    operator+(StrideT delta) const
        requires(InplaceAddable<JumpIterator, StrideT>)
    {
        auto copied = this;
        copied += delta;
        return copied;
    }

    JumpIterator&
    operator-=(StrideT delta)
        requires(std::bidirectional_iterator<Iter>)
    {
        if constexpr (InplaceSubtractable<Iter, StrideT>)
            wrappedIter -= delta;
        else
            std::advance(wrappedIter, -stride);

        return *this;
    }

    JumpIterator
    operator-(StrideT delta) const
        requires(InplaceSubtractable<JumpIterator, difference_type>)
    {
        auto copied = this;
        copied -= delta;
        return copied;
    }

    friend bool
    operator==(JumpIterator const& a, JumpIterator const& b) {
        return a.wrappedIter == b.wrappedIter;
    }

    friend bool
    operator!=(JumpIterator const& a, JumpIterator const& b) {
        return a.wrappedIter != b.wrappedIter;
    }

    friend bool
    operator<(JumpIterator const& a, JumpIterator const& b)
        requires(ComparableLT<Iter>)
    {
        return a.wrappedIter < b.wrappedIter;
    }

    friend bool
    operator<=(JumpIterator const& a, JumpIterator const& b)
        requires(ComparableLE<Iter>)
    {
        return a.wrappedIter <= b.wrappedIter;
    }

    friend bool
    operator>(JumpIterator const& a, JumpIterator const& b)
        requires(ComparableGT<Iter>)
    {
        return a.wrappedIter > b.wrappedIter;
    }

    friend bool
    operator>=(JumpIterator const& a, JumpIterator const& b)
        requires(ComparableGE<Iter>)
    {
        return a.wrappedIter >= b.wrappedIter;
    }

private:
    Iter    wrappedIter;
    StrideT stride;
};
