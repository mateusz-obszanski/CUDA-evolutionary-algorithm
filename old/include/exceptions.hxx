#pragma once

#include <stdexcept>
#include <string>

namespace exceptions {
    class NotImplementedError : public std::exception {
        const std::string msg;

    public:
        NotImplementedError(const std::string& msg) : msg{msg} {}

        const char*
        what() const noexcept override {
            return msg.c_str();
        }
    };
} // namespace exceptions
