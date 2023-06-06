#pragma once
#include "../matrix_utils.hxx"
#include "../permutation.hxx"

template <InversionVectorCoding CODING = InversionVectorCoding::SHORT>
class CoderInversionVec {
public:
    using GeneT = int;

    const int solutionLength;

    [[nodiscard]] CoderInversionVec(const int solutionLength) noexcept
    : solutionLength(solutionLength) {}

    void
    encode_many(std::vector<GeneT> const& solutions,
                std::vector<GeneT>&       out) const {
        const int nSolutions = solutions.size() / solutionLength;

        const MatrixView<const GeneT*> mxSolutions(solutions.data(), nSolutions,
                                                   solutionLength);
        const MatrixView mxEncoded(out.data(), nSolutions, encoded_length());

        for (int i{0}; i < nSolutions; ++i) {
            using IterOut = GeneT*;
            permutation_to_inversion_vector<GeneT, IterOut, CODING>(
                mxSolutions.get_row(i), mxEncoded.get_row(i).data());
        }
    }

    [[nodiscard]] std::vector<GeneT>
    encode_many(std::vector<GeneT> const& solutions) const {
        const int nSolutions = solutions.size() / solutionLength;

        std::vector<GeneT> out(nSolutions * encoded_length());
        encode_many(solutions, out);

        return out;
    }

    void
    decode_many(std::vector<GeneT> const& code, std::vector<GeneT>& out) const {
        const int nSolutions = code.size() / encoded_length();

        const MatrixView<const GeneT*> mxCode(code.data(), nSolutions,
                                              solutionLength);
        const MatrixView mxDecoded(out.data(), nSolutions, solutionLength);

        for (int i{0}; i < nSolutions; ++i) {
            using IterOut = GeneT*;
            inversion_vector_to_permutation<GeneT, IterOut, CODING>(
                mxCode.get_row(i), mxDecoded.get_row(i).data());
        }
    }

    [[nodiscard]] std::vector<GeneT>
    decode_many(std::vector<GeneT> const& code) const {
        const int nSolutions = code.size() / encoded_length();

        std::vector<int> decoded(nSolutions * encoded_length());
        decode_many(code, decoded.begin());

        return decoded;
    }

    [[nodiscard]] constexpr auto
    encoded_length() const noexcept {
        if constexpr (CODING == InversionVectorCoding::SHORT)
            return solutionLength - 1;
        else
            return solutionLength;
    }

    [[nodiscard]] constexpr auto
    decoded_length() const noexcept {
        return solutionLength;
    }
};
