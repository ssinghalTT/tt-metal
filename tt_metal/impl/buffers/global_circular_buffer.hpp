// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class Buffer;
class Device;

}  // namespace v0

namespace v1 {

namespace experimental {

class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(
        Device* device,
        const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type);

    GlobalCircularBuffer(const GlobalCircularBuffer&) = default;
    GlobalCircularBuffer& operator=(const GlobalCircularBuffer&) = default;

    GlobalCircularBuffer(GlobalCircularBuffer&&) noexcept = default;
    GlobalCircularBuffer& operator=(GlobalCircularBuffer&&) noexcept = default;

    static std::shared_ptr<GlobalCircularBuffer> create(
        Device* device,
        const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1);

    const Buffer& cb_buffer() const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    DeviceAddr buffer_address() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping() const;

    static constexpr auto attribute_names = std::forward_as_tuple("sender_receiver_core_mapping", "size");
    const auto attribute_values() const { return std::make_tuple(this->sender_receiver_core_mapping_, this->size_); }

private:
    void setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender);

    // GlobalCircularBuffer is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<Buffer> cb_buffer_;
    std::shared_ptr<Buffer> cb_config_buffer_;
    Device* device_;
    std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t size_ = 0;
};

}  // namespace experimental

}  // namespace v1

}  // namespace tt::tt_metal

template <>
struct fmt::formatter<std::unordered_map<CoreCoord, CoreRangeSet>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::unordered_map<CoreCoord, CoreRangeSet>& map, format_context& ctx) const
        -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << map;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

namespace std {

template <>
struct hash<std::unordered_map<CoreCoord, CoreRangeSet>> {
    std::size_t operator()(const std::unordered_map<CoreCoord, CoreRangeSet>& map) const {
        std::size_t seed = 0;
        for (const auto& kv : map) {
            tt::utils::hash_combine(seed, kv.first);
            tt::utils::hash_combine(seed, kv.second);
        }
        return seed;
    }
};

template <>
struct hash<tt::tt_metal::v1::experimental::GlobalCircularBuffer> {
    std::size_t operator()(const tt::tt_metal::v1::experimental::GlobalCircularBuffer& obj) const {
        std::size_t seed = 0;
        tt::utils::hash_combine(seed, obj.sender_receiver_core_mapping());
        tt::utils::hash_combine(seed, obj.size());
        return seed;
    }
};

}  // namespace std
