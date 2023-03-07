#pragma once

#include <stdexcept>

#define BASE_ERROR std::runtime_error

#define DEFINE_SIMPLE_ERROR(CLS_NAME, WHAT_MSG) \
    class CLS_NAME : public BASE_ERROR {        \
    public:                                     \
        const char*                             \
        what() const noexcept override {        \
            return #CLS_NAME ": " WHAT_MSG;     \
        }                                       \
    };
