// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::dispatch {

// Si KB Prefix
constexpr auto operator""_KB(const unsigned long long v) -> uint32_t { return 1024 * v; }

constexpr auto operator""_MB(const unsigned long long v) -> uint32_t { return 1024 * 1024 * v; }

constexpr auto operator""_GB(const unsigned long long v) -> uint32_t { return 1024 * 1024 * 1024 * v; }

inline uint32_t align_size(uint32_t sz, uint32_t alignment) {
    return ((sz + alignment - 1) / alignment * alignment);
}

template <typename T>
inline T align_addr(T addr, uint32_t alignment) {
    return ((addr - 1) | (alignment - 1)) + 1;
}

}  // namespace tt::tt_metal::dispatch
