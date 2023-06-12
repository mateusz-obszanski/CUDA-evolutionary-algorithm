#pragma once

#include "./text.hxx"
#include <source_location>
#include <stdexcept>
#include <string>

namespace errors {

namespace {

using std::source_location;

}

class NotImplementedException : public std::exception {
public:
    const source_location locInfo;
    const std::string     msg;

    NotImplementedException() = delete;
    NotImplementedException(source_location locInfo = source_location::current())
    : locInfo{locInfo}, msg{text::fmtLineInfo(locInfo)} {}

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

[[noreturn]] void
throwNotImplemented(source_location locInfo = source_location::current()) {
    throw NotImplementedException(locInfo);
}

} // namespace errors
