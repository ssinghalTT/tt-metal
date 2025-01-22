// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint8_t>,
    owned_buffer::Buffer<uint16_t>,
    owned_buffer::Buffer<int32_t>,
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>>;

// HostDataType supports all types included in OwnedBuffer as well as void*
static_assert(
    std::variant_size_v<OwnedBuffer> + 1 == std::variant_size_v<tt::tt_metal::HostDataType>,
    "The data types supported in OwnedBuffer must match those in HostDataType.");
struct OwnedStorage {
    OwnedBuffer buffer;
    OwnedStorage() = default;
    OwnedStorage(OwnedBuffer buffer_) : buffer(std::move(buffer_)) {}

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    inline void insert_buffer(const OwnedBuffer& buffer_) { this->buffer = buffer_; }

    inline OwnedBuffer get_buffer() const { return this->buffer; }

    inline bool is_allocated() const {
        return std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, buffer);
    }
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;
    DeviceStorage() = default;
    DeviceStorage(DeviceBuffer buffer_) : buffer(std::move(buffer_)) {}

    const MemoryConfig memory_config() const {
        if (this->buffer.get() == nullptr) {
            TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
        }

        std::optional<ShardSpec> shard_spec = std::nullopt;
        if (is_sharded(this->buffer->buffer_layout())) {
            shard_spec = this->buffer->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = this->buffer->buffer_layout(),
            .buffer_type = this->buffer->buffer_type(),
            .shard_spec = shard_spec};
    }

    inline void insert_buffer(DeviceBuffer buffer_) { this->buffer = buffer_; }

    inline DeviceBuffer get_buffer() const { return this->buffer; }
    static constexpr auto attribute_names = std::forward_as_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(this->memory_config()); }

    inline bool is_allocated() const { return buffer && buffer->is_allocated(); }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint8_t>,
    borrowed_buffer::Buffer<uint16_t>,
    borrowed_buffer::Buffer<int32_t>,
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>>;
struct BorrowedStorage {
    BorrowedBuffer buffer;

    BorrowedStorage() = default;
    std::function<void()> on_creation_callback = [] {};
    std::function<void()> on_destruction_callback = [] {};

    explicit BorrowedStorage(
        const BorrowedBuffer& buffer,
        const std::function<void()>& on_creation_callback,
        const std::function<void()>& on_destruction_callback) :
        buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage& other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage&& other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
    }

    BorrowedStorage operator=(BorrowedStorage&& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
        return *this;
    }

    ~BorrowedStorage() { this->on_destruction_callback(); }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    inline bool is_allocated() const { return true; }
};

struct MultiDeviceHostStorage {
    DistributedTensorConfig strategy;
    std::vector<OwnedBuffer> buffers;
    std::vector<TensorSpec> specs;
    mutable std::mutex mtx;

    friend void swap(MultiDeviceHostStorage& first, MultiDeviceHostStorage& second) {
        std::scoped_lock lock(first.mtx, second.mtx);
        // enable ADL (not necessary, but good practice)
        using std::swap;

        swap(first.strategy, second.strategy);
        swap(first.buffers, second.buffers);
        swap(first.specs, second.specs);
    }

    MultiDeviceHostStorage() = default;
    MultiDeviceHostStorage(
        DistributedTensorConfig strategy_, std::vector<OwnedBuffer> buffers_, std::vector<TensorSpec> specs_) :
        strategy(strategy_), buffers(buffers_), specs(specs_) {}
    MultiDeviceHostStorage(MultiDeviceHostStorage&& other) { swap(*this, other); }
    // unfotunately we need to have this code written manually.
    MultiDeviceHostStorage(const MultiDeviceHostStorage& other) {
        std::scoped_lock lock(other.mtx);
        strategy = other.strategy;
        buffers = other.buffers;
        specs = other.specs;
    }

    MultiDeviceHostStorage& operator=(const MultiDeviceHostStorage& other) {
        MultiDeviceHostStorage temp(other);
        swap(*this, temp);
        return *this;
    }

    MultiDeviceHostStorage& operator=(MultiDeviceHostStorage&& other) {
        swap(*this, other);
        return *this;
    }

    bool operator==(const MultiDeviceHostStorage& other) {
        return this->strategy == other.strategy and this->buffers == other.buffers and this->specs == other.specs;
    }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
    // preinitialize empty tensor handles and use/populate them in the worker threads.
    void insert_buffer_and_spec_for_device(int buffer_index, const OwnedBuffer& buffer, TensorSpec spec) {
        std::lock_guard<std::mutex> lock(mtx);
        buffers[buffer_index] = buffer;
        specs[buffer_index] = std::move(spec);
    }

    OwnedBuffer get_buffer(int buffer_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_ASSERT(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    OwnedBuffer& get_buffer(int buffer_index) {
        std::lock_guard<std::mutex> lock(mtx);
        TT_ASSERT(buffer_index < buffers.size(), "Buffer not found for buffer_index {}", buffer_index);
        return buffers[buffer_index];
    }

    TensorSpec get_tensor_spec(int spec_index) const {
        std::lock_guard<std::mutex> lock(mtx);
        TT_ASSERT(spec_index < specs.size(), "Buffer not found for device {}", spec_index);
        return specs[spec_index];
    }

    uint32_t num_buffers() const {
        std::lock_guard<std::mutex> lock(mtx);
        return buffers.size();
    }

    inline bool is_allocated() const {
        // not sure what is better mutex for each buffer 10 times or one here.
        // I think this one is better.
        std::lock_guard<std::mutex> lock(mtx);

        return std::all_of(buffers.begin(), buffers.end(), [](auto&& buffer) {
            return std::visit([](auto&& buffer) -> bool { return buffer.is_allocated(); }, buffer);
        });
    }
};

struct MultiDeviceStorage {
    DistributedTensorConfig strategy;
    std::vector<int> ordered_device_ids;
    std::unordered_map<int, DeviceBuffer> buffers;
    std::unordered_map<int, TensorSpec> specs;
    mutable std::mutex buffer_mtx;
    mutable std::mutex shape_mtx;
    MultiDeviceStorage() = default;

    friend void swap(MultiDeviceStorage& first, MultiDeviceStorage& second) {
        std::scoped_lock lock(first.buffer_mtx, first.shape_mtx, second.buffer_mtx, second.shape_mtx);

        swap(first.strategy, second.strategy);
        swap(first.ordered_device_ids, second.ordered_device_ids);
        swap(first.buffers, second.buffers);
        swap(first.specs, second.specs);
    }

    MultiDeviceStorage(
        DistributedTensorConfig strategy_,
        std::vector<int> ordered_device_ids_,
        std::unordered_map<int, DeviceBuffer> buffers_,
        std::unordered_map<int, TensorSpec> specs_) :
        strategy(std::move(strategy_)),
        ordered_device_ids(std::move(ordered_device_ids_)),
        buffers(std::move(buffers_)),
        specs(std::move(specs_)) {}

    MultiDeviceStorage(MultiDeviceStorage&& other) { swap(*this, other); }

    MultiDeviceStorage(const MultiDeviceStorage& other) {
        std::scoped_lock lock(other.buffer_mtx, other.shape_mtx);
        ordered_device_ids = other.ordered_device_ids;
        strategy = other.strategy;
        buffers = other.buffers;
        specs = other.specs;
    }

    MultiDeviceStorage& operator=(const MultiDeviceStorage& other) {
        MultiDeviceStorage tmp(other);
        swap(*this, tmp);
        return *this;
    }

    MultiDeviceStorage& operator=(MultiDeviceStorage&& other) {
        swap(*this, other);
        return *this;
    }

    bool operator==(const MultiDeviceStorage& other) {
        return this->ordered_device_ids == other.ordered_device_ids and this->strategy == other.strategy and
               this->buffers == other.buffers and this->specs == other.specs;
    }

    inline const MemoryConfig memory_config() const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        TT_FATAL(
            !this->ordered_device_ids.empty(), "No device ids in list. Please ensure fields are initialized properly.");
        auto first_device_id = this->ordered_device_ids[0];
        if (this->buffers.at(first_device_id).get() == nullptr) {
            TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
        }
        std::optional<ShardSpec> shard_spec = std::nullopt;
        if (is_sharded(this->buffers.at(first_device_id)->buffer_layout())) {
            shard_spec = this->buffers.at(first_device_id)->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = this->buffers.at(first_device_id)->buffer_layout(),
            .buffer_type = this->buffers.at(first_device_id)->buffer_type(),
            .shard_spec = shard_spec};
    }

    static constexpr auto attribute_names = std::forward_as_tuple();
    const auto attribute_values() const { return std::forward_as_tuple(); }

    // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
    // preinitialize empty tensor handles and use/populate them in the worker threads.
    std::vector<DeviceBuffer> get_buffers() const;

    inline void insert_buffer_and_spec_for_device(IDevice* device, const DeviceBuffer buffer, TensorSpec spec) {
        std::scoped_lock lock(buffer_mtx, shape_mtx);
        TT_ASSERT(
            device == buffer->device(),
            "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
        buffers.insert({device->id(), buffer});
        specs.insert({device->id(), std::move(spec)});
    }

    inline DeviceBuffer get_buffer_for_device(IDevice* device) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        TT_ASSERT(buffers.find(device->id()) != buffers.end(), "Buffer not found for device {}", device->id());
        TT_ASSERT(
            buffers.at(device->id())->device() == device,
            "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
        return buffers.at(device->id());
    }

    inline DeviceBuffer& get_buffer_for_device(IDevice* device) {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        TT_ASSERT(buffers.find(device->id()) != buffers.end(), "Buffer not found for device {}", device->id());
        TT_ASSERT(
            buffers.at(device->id())->device() == device,
            "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
        return buffers.at(device->id());
    }

    inline DeviceBuffer get_buffer_for_device_id(uint32_t device_id) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.at(device_id);
    }

    inline TensorSpec get_tensor_spec_for_device(IDevice* device) const {
        std::lock_guard<std::mutex> lock(shape_mtx);
        TT_ASSERT(specs.find(device->id()) != specs.end(), "Shape not found for device {}", device->id());
        return specs.at(device->id());
    }

    inline uint32_t num_buffers() const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.size();
    }

    inline bool has_buffer_for_device(IDevice* device) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.find(device->id()) != buffers.end();
    }

    inline bool has_buffer_for_device_id(uint32_t device_id) const {
        std::lock_guard<std::mutex> lock(buffer_mtx);
        return buffers.find(device_id) != buffers.end();
    }

    inline bool is_allocated() const {
        std::lock_guard<std::mutex> lock(buffer_mtx);

        return std::all_of(
            ordered_device_ids.begin(), ordered_device_ids.end(), [&buffers = this->buffers](auto&& device_id) {
                const auto& buffer = buffers.at(device_id);
                return buffer && buffer->is_allocated();
            });
    }
};

using Storage = std::variant<OwnedStorage, DeviceStorage, BorrowedStorage, MultiDeviceHostStorage, MultiDeviceStorage>;

template <typename T>
constexpr void raise_unsupported_storage() {
    static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported Storage");
}

}  // namespace tt::tt_metal
