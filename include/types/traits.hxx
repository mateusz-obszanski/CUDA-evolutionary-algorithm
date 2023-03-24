#pragma once

#include <functional>

namespace traits {

template <typename Callable>
using return_type_of_t =
    typename decltype(std::function{std::declval<Callable>()})::result_type;

}
