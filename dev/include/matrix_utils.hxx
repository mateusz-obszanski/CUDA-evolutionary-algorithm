#pragma once
#include "./io_utils.hxx"
#include <iostream>

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

template <typename T, typename IterT = T*>
class MatrixView {
public:
    using value_type = T;
    using pointer    = IterT;
    using Size       = std::pair<int, int>;

    [[nodiscard]] MatrixView(pointer p, const Size size) : mMemory{p}, mSize{size} {};
    [[nodiscard]] MatrixView(pointer p, const int n) : MatrixView(p, {n, n}) {}

    [[nodiscard]] T
    get(const int row, const int col) const {
        return mMemory[linear_idx(row, col)];
    }

    void
    set(const int row, const int col, const T elem) {
        mMemory[linear_idx(row, col)] = elem;
    }

    [[nodiscard]] int
    width() const noexcept {
        return mSize.second;
    }

    [[nodiscard]] int
    height() const noexcept {
        return mSize.second;
    }

private:
    pointer mMemory;
    Size    mSize;

    [[nodiscard]] int
    linear_idx(const int row, const int col) const noexcept {
        return row * width() + col;
    }
};

inline auto&
printColumnIdxs(
    const int         ncols,
    const std::size_t maxElemWidth,
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
prettyPrintMx(Iter begin, const int nrows, const int ncols) {
    const std::string elemSeparator{" "};
    const std::string idxColSep{" | "};

    const auto nElems = nrows * ncols;
    const auto end    = begin + nElems;

    const auto strElems    = stringify_many(begin, end);
    const auto maxStrWidth = std::max_element(strElems.cbegin(), strElems.cend(), cmpStrByLength)->length();

    using ElemT = typename Iter::value_type;
    const MatrixView<ElemT, Iter> mx(begin, {nrows, ncols});

    // print header with column indices
    const int maxRowIdxWidth = (nrows - 1) / 10 + 1;

    const auto columnHeaderPadding = maxRowIdxWidth + idxColSep.length();

    printColumnIdxs(ncols, maxStrWidth, columnHeaderPadding, elemSeparator.length());
    printSeparationLine(columnHeaderPadding + ncols * (maxStrWidth + elemSeparator.length()));

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

template <typename T>
inline void
prettyPrintMx(std::vector<std::vector<T>>& mx) {
    const std::string elemSeparator{" "};
    const std::string idxColSep{" | "};

    const int nrows = mx.size();
    const int ncols = mx[0].size();

    std::size_t maxStrWidth = 0;

    for (const auto& row : mx) {
        const auto strElems       = stringify_many(row.cbegin(), row.cend());
        const auto maxRowStrWidth = std::max_element(strElems.cbegin(), strElems.cend(), cmpStrByLength)->length();

        if (maxStrWidth < maxRowStrWidth)
            maxStrWidth = maxRowStrWidth;
    }

    // print header with column indices
    const int maxRowIdxWidth = (nrows - 1) / 10 + 1;

    const auto columnHeaderPadding = maxRowIdxWidth + idxColSep.length();

    printColumnIdxs(ncols, maxStrWidth, columnHeaderPadding, elemSeparator.length());
    printSeparationLine(columnHeaderPadding + ncols * (maxStrWidth + elemSeparator.length()));

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
