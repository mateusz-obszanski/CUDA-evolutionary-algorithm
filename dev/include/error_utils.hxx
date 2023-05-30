#pragma once
#include <stdexcept>

class DeprecationError : public std::exception {
public:
    DeprecationError(const std::string& functionality = "<unknown>") : msg{"Deprecation error: " + functionality} {}

    const char*
    what() const noexcept override {
        return msg.c_str();
    }

private:
    const std::string msg;
};
