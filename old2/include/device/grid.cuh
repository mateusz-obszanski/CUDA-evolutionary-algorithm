#pragma once

namespace device {
namespace grid {

enum class DimN {
    NONE    = 0,
    D1      = 1,
    D2      = 2,
    D3      = 3,
    UNKNOWN = D3, // D3 will work with every dimension, so it is a safe fallback
};

}
} // namespace device
