#pragma once
#include "./io_utils.hxx"
#include "./jump_iterator.hxx"
#include "common_concepts.hxx"
#include <bits/iterator_concepts.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>

template <typename Iter>
inline void
printMx(Iter begin, const int nrows, const int ncols) {
    std::cout << "[\n";

    for (int row{0}; row < nrows; ++row) {
        printIter(begin, begin + ncols);
        begin += ncols;
    }

    std::cout << "]\n";
}

namespace {

template <std::contiguous_iterator Iter> struct MatrixRowIter {
private:
    using RowIter       = JumpIterator<Iter>;
    using RowIterTraits = std::iterator_traits<RowIter>;
    using T             = typename RowIterTraits::value_type;

public:
    using iterator_category = typename RowIter::iterator_category;
    using difference_type   = typename RowIter::difference_type;
    using value_type        = std::span<T>;
    using pointer           = std::unique_ptr<value_type>;
    using reference         = value_type;

    MatrixRowIter() = delete;
    MatrixRowIter(Iter data, int rowLength) : row(data, rowLength) {}

    reference
    operator*() {
        return {static_cast<Iter>(row),
                static_cast<std::size_t>(std::abs(row.jump_length()))};
    }

    pointer
    operator->() {
        return std::make_unique<value_type>(*row, row.jump_length());
    }

    MatrixRowIter&
    operator++()
        requires(PreIncrementable<RowIter>)
    {
        ++row;
        return *this;
    }

    MatrixRowIter
    operator++(int)
        requires(PreIncrementable<MatrixRowIter> and
                 std::copyable<MatrixRowIter>)
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    MatrixRowIter&
    operator--()
        requires(PreDecrementable<RowIter>)
    {
        --row;
        return *this;
    }

    MatrixRowIter
    operator--(int)
        requires(PreDecrementable<MatrixRowIter> and
                 std::copyable<MatrixRowIter>)
    {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    MatrixRowIter&
    operator+=(difference_type delta)
        requires(InplaceAddable<RowIter, difference_type>)
    {
        row += delta;
        return *this;
    }

    MatrixRowIter
    operator+(difference_type delta) const
        requires(InplaceAddable<MatrixRowIter, difference_type>)
    {
        auto copied = *this;
        copied += delta;
        return copied;
    }

    MatrixRowIter&
    operator-=(difference_type delta)
        requires(InplaceSubtractable<RowIter, difference_type>)
    {
        row -= delta;
        return *this;
    }

    MatrixRowIter
    operator-(difference_type delta) const
        requires(InplaceSubtractable<MatrixRowIter, difference_type>)
    {
        auto copied = this;
        copied -= delta;
        return copied;
    }

    friend bool
    operator==(MatrixRowIter const& a, MatrixRowIter const& b) {
        return a.row == b.row;
    }

    friend bool
    operator!=(MatrixRowIter const& a, MatrixRowIter const& b) {
        return a.row != b.row;
    }

    friend bool
    operator<(MatrixRowIter const& a, MatrixRowIter const& b)
        requires(ComparableLT<RowIter>)
    {
        return a.row < b.row;
    }

    friend bool
    operator<=(MatrixRowIter const& a, MatrixRowIter const& b)
        requires(ComparableLE<RowIter>)
    {
        return a.row <= b.row;
    }

    friend bool
    operator>(MatrixRowIter const& a, MatrixRowIter const& b)
        requires(ComparableGT<RowIter>)
    {
        return a.row > b.row;
    }

    friend bool
    operator>=(MatrixRowIter const& a, MatrixRowIter const& b)
        requires(ComparableGE<RowIter>)
    {
        return a.row >= b.row;
    }

private:
    RowIter row;
};

template <std::contiguous_iterator Iter> struct MatrixViewRowIterProxy {
    MatrixViewRowIterProxy() = delete;
    [[nodiscard]] MatrixViewRowIterProxy(Iter ptrData, int nrows, int rowLength)
    : ptrData(ptrData), nrows(nrows), rowLength(rowLength) {}

    auto
    begin() {
        return MatrixRowIter<Iter>(ptrData, rowLength);
    }

    auto
    end() {
        return begin() + nrows;
    }

private:
    Iter const ptrData;
    const int  nrows;
    const int  rowLength;
};

template <typename Iter> struct MatrixViewColIterProxy {
    MatrixViewColIterProxy() = delete;
    [[nodiscard]] MatrixViewColIterProxy(Iter ptrData, int ncols)
    : ptrData(ptrData), ncols(ncols) {}

    auto
    begin() {
        return JumpIterator(JumpIterator(ptrData, ncols), 1);
    }

    auto
    end() {
        return begin() + ncols;
    }

private:
    const Iter ptrData;
    const int  ncols;
};

} // namespace

template <typename IterT> class MatrixView {
public:
    using pointer    = IterT;
    using value_type = std::iterator_traits<IterT>::value_type;
    using Size       = std::pair<int, int>;
    using View       = std::span<value_type>;

    [[nodiscard]] MatrixView(pointer p, const Size size)
    : mMemory{p}, mSize{size} {};
    [[nodiscard]] MatrixView(pointer p, const int n) : MatrixView(p, {n, n}) {}
    [[nodiscard]] MatrixView(pointer p, const int nrows, const int ncols)
    : MatrixView(p, {nrows, ncols}) {}

    [[nodiscard]] pointer
    data() const {
        return mMemory;
    }

    [[nodiscard]] value_type
    get(const int row, const int col) const {
        return mMemory[linear_idx(row, col)];
    }

    void
    set(const int row, const int col, const value_type elem) {
        mMemory[linear_idx(row, col)] = elem;
    }

    [[nodiscard]] View
    get_row(const int i) const noexcept {
        return {row_begin(i), static_cast<unsigned int>(width())};
    }

    [[nodiscard]] int
    width() const noexcept {
        return mSize.second;
    }

    [[nodiscard]] int
    height() const noexcept {
        return mSize.first;
    }

    [[nodiscard]] auto
    shape() const {
        return mSize;
    }

    void
    print() const {
        printMx(mMemory, mSize.first, mSize.second);
    }

    void
    pretty_print() const {
        prettyPrintMx(mMemory, mSize.first, mSize.second);
    }

    auto
    rows()
        requires(std::contiguous_iterator<IterT>)
    {
        return MatrixViewRowIterProxy<IterT>(mMemory, mSize.first,
                                             mSize.second);
    }

private:
    pointer mMemory;
    Size    mSize;

    [[nodiscard]] int
    row_offset(const int i) const noexcept {
        return i * width();
    }

    [[nodiscard]] int
    linear_idx(const int row, const int col) const noexcept {
        return row_offset(row) + col;
    }

    [[nodiscard]] pointer
    row_begin(const int i) const noexcept {
        return mMemory + row_offset(i);
    }
};

inline auto&
printColumnIdxs(const int ncols, const std::size_t maxElemWidth,
                const std::size_t lpadding        = 0,
                const std::size_t separatorLength = 1,
                std::ostream&     out             = std::cout) {

    const std::size_t maxColIdxWidth = (ncols - 1) / 10 + 1;
    const std::size_t maxColWidth    = std::max(maxColIdxWidth, maxElemWidth);

    // additional padding by index column length
    printPadding(lpadding);

    for (int i{0}; i < ncols; ++i)
        out << to_lpad(maxColWidth, i) << spaces(separatorLength);

    return out;
}

template <typename Iter>
inline void
prettyPrintMx(MatrixView<Iter> const& mx) {
    const auto nrows = mx.height();
    const auto ncols = mx.width();
    const auto begin = mx.data();

    const std::string elemSeparator{" "};
    const std::string idxColSep{" | "};

    const auto nElems = nrows * ncols;
    const auto end    = begin + nElems;

    const auto strElems = stringify_many(begin, end);
    const auto maxStrWidth =
        std::max_element(strElems.cbegin(), strElems.cend(), cmpStrByLength)
            ->length();

    // print header with column indices
    const int maxRowIdxWidth = (nrows - 1) / 10 + 1;

    const auto columnHeaderPadding = maxRowIdxWidth + idxColSep.length();

    printColumnIdxs(ncols, maxStrWidth, columnHeaderPadding,
                    elemSeparator.length());
    printSeparationLine(columnHeaderPadding +
                        ncols * (maxStrWidth + elemSeparator.length()));

    const auto printIndexedRow = [&](const int row) {
        const auto printElem = [=, &mx](const int row, const int col) {
            printPadded(maxStrWidth, mx.get(row, col));
        };

        // print current row index
        const auto rowIdxColWidth = maxRowIdxWidth;
        printPadded(rowIdxColWidth, row);
        std::cout << idxColSep;

        // print row of matrix elements, excluding the last one
        for (int col{0}; col < ncols - 1; ++col) {
            printElem(row, col);
            std::cout << elemSeparator;
        }

        // print last element without sep at the end
        const auto lastCol = ncols - 1;
        printElem(row, lastCol);
    };

    for (int i{0}; i < nrows; ++i) {
        printIndexedRow(i);
        std::cout << '\n';
    }
}

template <typename Iter>
inline void
prettyPrintMx(Iter begin, const int nrows, const int ncols) {
    const MatrixView mx(begin, {nrows, ncols});
    prettyPrintMx(mx);
}

template <typename T>
inline void
prettyPrintMx(std::vector<std::vector<T>>& mx) {
    const std::string elemSeparator{" "};
    const std::string idxColSep{" | "};

    const int nrows = mx.size();
    const int ncols = mx[0].size();

    std::size_t maxStrWidth = 0;

    for (const auto& row : mx) {
        const auto strElems = stringify_many(row.cbegin(), row.cend());
        const auto maxRowStrWidth =
            std::max_element(strElems.cbegin(), strElems.cend(), cmpStrByLength)
                ->length();

        if (maxStrWidth < maxRowStrWidth)
            maxStrWidth = maxRowStrWidth;
    }

    // print header with column indices
    const int maxRowIdxWidth = (nrows - 1) / 10 + 1;

    const auto columnHeaderPadding = maxRowIdxWidth + idxColSep.length();

    printColumnIdxs(ncols, maxStrWidth, columnHeaderPadding,
                    elemSeparator.length());
    printSeparationLine(columnHeaderPadding +
                        ncols * (maxStrWidth + elemSeparator.length()));

    const auto printIndexedRow = [&](const int row) {
        const auto printElem = [=, &mx](const int row, const int col) {
            printPadded(maxStrWidth, mx[row][col]);
        };

        // print current row index
        const auto rowIdxColWidth = maxRowIdxWidth;
        printPadded(rowIdxColWidth, row);
        std::cout << idxColSep;

        // print row of matrix elements, excluding the last one
        for (int col{0}; col < ncols - 1; ++col) {
            printElem(row, col);
            std::cout << elemSeparator;
        }

        // print last element without sep at the end
        const auto lastCol = ncols - 1;
        printElem(row, lastCol);
    };

    for (int i{0}; i < nrows; ++i) {
        printIndexedRow(i);
        std::cout << '\n';
    }
}

template <typename Iter>
inline void
prettyPrintMx(Iter begin, const int n) {
    prettyPrintMx(begin, n, n);
}

/// print square matrix
template <typename Iter>
inline void
printMx(Iter begin, const std::size_t n) {
    printMx(begin, n, n);
}
