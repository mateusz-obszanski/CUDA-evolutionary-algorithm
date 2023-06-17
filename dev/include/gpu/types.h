#pragma once

template <typename T>
using device_ptr = T*;

template <typename T>
using const_device_ptr = const T*;

template <typename T>
using device_ptr_in = const T* const;

template <typename T>
using device_ptr_out = T* const;

template <typename T>
using device_ptr_inout = T* const;
