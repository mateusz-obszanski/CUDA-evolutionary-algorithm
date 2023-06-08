#pragma once
#include <concepts>
#include <iterator>
#include <sstream>
#include <string>

template <typename T>
concept IsConst = std::is_const_v<T>;

template <typename Iter>
concept IsConstIter = IsConst<typename std::iterator_traits<Iter>::value_type>;

template <typename T1, typename T2 = T1>
concept ComparableLT = requires(T1 x, T2 y) {
    { x < y } -> std::convertible_to<bool>;
};

template <typename T1, typename T2 = T1>
concept ComparableLE = requires(T1 x, T2 y) {
    { x <= y } -> std::convertible_to<bool>;
};

template <typename T1, typename T2 = T1>
concept ComparableGT = requires(T1 x, T2 y) {
    { x > y } -> std::convertible_to<bool>;
};

template <typename T1, typename T2 = T1>
concept ComparableGE = requires(T1 x, T2 y) {
    { x >= y } -> std::convertible_to<bool>;
};

template <typename T1, typename T2 = T1>
concept ComparableEQ = requires(T1 x, T2 y) {
    { x == y } -> std::convertible_to<bool>;
};

template <typename T1, typename T2 = T1>
concept ComparableNE = requires(T1 x, T2 y) {
    { x != y } -> std::convertible_to<bool>;
};

template <typename T>
concept PreIncrementable = requires(T x) {
    { ++x } -> std::same_as<T&>;
};

template <typename T>
concept PostIncrementable = requires(T x) {
    { x++ } -> std::same_as<T>;
};

template <typename T>
concept PreDecrementable = requires(T x) {
    { --x } -> std::same_as<T&>;
};

template <typename T>
concept PostDecrementable = requires(T x) {
    { x-- } -> std::same_as<T>;
};

template <typename T1, typename T2 = T1, typename R = T1>
concept Addable = requires(T1 x, T2 y) {
    { x + y } -> std::same_as<R>;
};

template <typename T1, typename T2 = T1, typename R = T1>
concept Subtractable = requires(T1 x, T2 y) {
    { x - y } -> std::same_as<R>;
};

template <typename T1, typename T2 = T1>
concept InplaceAddable = requires(T1 x, T2 y) { x += y; };

template <typename T1, typename T2 = T1>
concept InplaceSubtractable = requires(T1 x, T2 y) { x -= y; };

template <typename T>
concept TriviallyStringifiable = requires(const T& x) { std::to_string(x); };

template <typename T>
concept SStreamStringifiable =
    requires(T const& x, std::ostringstream& stream) { (stream << x).str(); };

template <typename Fn, typename... Args>
concept NothrowInvocable = std::is_nothrow_invocable_v<Fn, Args...>;

template <typename T>
concept PrimitiveType = std::is_fundamental_v<T>;

template <typename Iter, typename T>
concept IterValConvertibleTo =
    std::convertible_to<typename std::iterator_traits<Iter>::value_type, T>;
