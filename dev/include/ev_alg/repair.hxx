#pragma once
#include "../permutation.hxx"

// class RepairNoOp {
//     template <typename Iter>
//     void
//     operator()(Iter, Iter) {
//     }
// };

/// repairing should be done on encoded solution
class RepairInversionVec {
    template <typename Iter>
    void
    operator()(Iter begin, Iter end) {
        repair_inversion_vector(begin, end);
    }
};
