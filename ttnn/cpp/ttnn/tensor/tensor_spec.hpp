// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal {

class TensorSpec final {
public:
    TensorSpec(ttnn::SimpleShape logical_shape, TensorLayout tensor_layout) :
        logical_shape_(std::move(logical_shape)),
        tensor_layout_(std::move(tensor_layout)),
        cached_padded_shape_(tensor_layout_.compute_padded_shape(logical_shape_)),
        cached_logical_2d_shape_(tensor_layout_.compute_logical_2d_shape(logical_shape_)),
        cached_physical_shape_(tensor_layout_.compute_physical_shape(logical_shape_)) {}
    TensorSpec(TensorSpec&&) noexcept = default;
    TensorSpec& operator=(TensorSpec&&) = default;
    TensorSpec(const TensorSpec&) = default;
    TensorSpec& operator=(const TensorSpec&) = default;
    bool operator==(const TensorSpec&) const = default;
    bool operator!=(const TensorSpec&) const = default;

    const ttnn::SimpleShape& logical_shape() const { return logical_shape_; }
    const TensorLayout& tensor_layout() const { return tensor_layout_; }
    DataType data_type() const { return tensor_layout_.get_data_type(); }
    Layout layout() const { return tensor_layout_.get_layout(); }
    PageConfig page_config() const { return tensor_layout_.get_page_config(); }
    const MemoryConfig& memory_config() const { return tensor_layout_.get_memory_config(); }
    const ttnn::SimpleShape& padded_shape() const { return cached_padded_shape_; }
    const Shape2D& logical_2d_shape() const { return cached_logical_2d_shape_; }
    const Shape2D& physical_shape() const { return cached_physical_shape_; }
    ttnn::Shape shape() const { return ttnn::Shape(logical_shape_.view(), cached_padded_shape_.view()); }

    Tile tile() const { return tensor_layout_.get_page_config().get_tile(); }

    Strides compute_strides() const { return tensor_layout_.compute_strides(logical_shape_); }
    std::optional<ShardSpecBuffer> compute_shard_spec_buffer() const {
        return tensor_layout_.compute_shard_spec_buffer(logical_shape_);
    }
    size_t compute_packed_buffer_size_bytes() const {
        return tensor_layout_.compute_packed_buffer_size_bytes(logical_shape_);
    }
    size_t compute_page_size_bytes() const { return tensor_layout_.compute_page_size_bytes(logical_shape_); }

    static constexpr auto attribute_names = std::forward_as_tuple("logical_shape", "tensor_layout");
    const auto attribute_values() const { return std::forward_as_tuple(logical_shape_, tensor_layout_); }

private:
    ttnn::SimpleShape logical_shape_;
    TensorLayout tensor_layout_;

    ttnn::SimpleShape cached_padded_shape_;
    Shape2D cached_logical_2d_shape_;
    Shape2D cached_physical_shape_;
};

}  // namespace tt::tt_metal
