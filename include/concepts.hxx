#pragma once

#include <concepts>

template <typename F, typename X, typename Y = X>
concept MappingFn = std::is_invocable_r<Y, F, X>::value;

template <typename F, typename X>
concept Endomorphism = MappingFn<F, X, X>;
