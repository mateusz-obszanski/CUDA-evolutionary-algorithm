#pragma once
#include "./common_concepts.hxx"
#include <concepts>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

template <typename T>
[[nodiscard]] inline constexpr T
square(T x) {
    return x * x;
}

template <typename T>
[[nodiscard]] inline constexpr T
isEven(T x) {
    return x % 2 == 0;
}

template <typename Iter>
inline auto
iter_to_ptr_cast(Iter iter) {
    return &(*iter);
}

inline auto
get_home_path() {
    // on Linux
    return std::filesystem::path(std::getenv("HOME"));
}

inline std::string
get_time_str(std::string const& fmt = "%d-%m-%Y %H-%M-%S") {
    const auto         t  = std::time(nullptr);
    const auto         tm = *std::localtime(&t);
    std::ostringstream ss;
    ss << std::put_time(&tm, fmt.c_str());
    return ss.str();
}

enum class CleanupFailMode { EXIT, PASS };

template <std::invocable Fn, CleanupFailMode MODE = CleanupFailMode::EXIT>
struct Cleanup {
    [[nodiscard]] Cleanup() noexcept(Fn()) = default;
    [[nodiscard]] Cleanup(Fn fn) : fn(fn){};

    ~Cleanup() noexcept
        requires(NothrowInvocable<Fn>)
    {
        fn();
    }

    ~Cleanup() noexcept
        requires(not NothrowInvocable<Fn>)
    {
        try {
            fn();
        } catch (...) {
            if constexpr (MODE == CleanupFailMode::EXIT)
                std::exit(1);
        }
    }

private:
    Fn fn;
};
