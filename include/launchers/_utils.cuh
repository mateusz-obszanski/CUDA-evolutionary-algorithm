#pragma once

namespace launcher {
    namespace utils {
        template <typename Size = std::size_t>
        PairOf<Size>
        calcLaunchParams1D(const Size nElems) {
            constexpr std::size_t blockSize = 1024;
            const auto            nBlocks   = cuda_utils::divCeil<std::size_t>(nElems, blockSize);

            return {nBlocks, blockSize};
        }
    } // namespace utils
} // namespace launcher
