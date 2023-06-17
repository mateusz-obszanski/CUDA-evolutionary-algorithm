#pragma once

#include <stdexcept>
#include <string>

namespace errors {

class NotImplementedException : public std::exception {
public:
    const int         lineNum;
    const std::string msg;

    NotImplementedException() = delete;
    NotImplementedException(int lineNum = -1)
    : lineNum(lineNum), msg{std::to_string(lineNum)} {}

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

[[noreturn]] void
throwNotImplemented(const int lineNum = -1) {
    throw NotImplementedException(lineNum);
}

} // namespace errors
