// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/wait.h>

#include <cstdint>
#include <cstdlib>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"

using tt::tt_metal::Tensor;

namespace ttnn::operations::sliding_window {

// Calculate Convolution on padded input buffer.
owned_buffer::Buffer<bfloat16> ref_conv_op(
    const Tensor& input_padded_tensor,
    const ttnn::SimpleShape& input_nchw_shape,
    uint32_t stride_h,
    uint32_t stride_w,
    const std::vector<float>& filter_vector,
    const ttnn::SimpleShape& filter_pyt_tensor_shape,
    const ttnn::SimpleShape& out_golden_pyt_tensor_shape);

// Calculate convolution using op_trace_metadata on padded input buffer.
owned_buffer::Buffer<bfloat16> conv_using_op_trace_metadata(
    const owned_buffer::Buffer<bfloat16>& input_padded_tensor_buf,
    const std::vector<float>& filter_vector,
    const std::vector<uint32_t>& op_trace_metadata,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t padded_input_w,
    uint32_t out_tensor_size);

// Calculate convolution using shards on padded input buffer.
owned_buffer::Buffer<bfloat16> conv_using_shard_boundaries(
    const owned_buffer::Buffer<bfloat16>& input_padded_tensor_buf,
    const std::vector<float>& filter_vector,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t padded_input_h,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t output_h,
    uint32_t output_w,
    uint32_t out_tensor_size);

// Calculate convolution using sliding window op configs on padded input buffer.
owned_buffer::Buffer<bfloat16> conv_using_sliding_window_op_config(
    const owned_buffer::Buffer<bfloat16>& input_padded_tensor_buf,
    const std::vector<float>& filter_vector,
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries,
    const std::vector<std::vector<uint16_t>>& sharded_input_top_left_indices,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t out_tensor_size);

// Calculate Padding using tensor metadata.
std::vector<bool> pad_metadata_from_tensor_metadata(const std::vector<std::pair<bool, uint32_pair_t>>& tensor_metadata);

// Calculate Indices of pads in padded input buffer using halo kernel config's flattened pad config.
std::vector<uint32_t> pad_indices_from_flattened_pad_config(
    const std::vector<std::vector<uint16_t>>& flattened_pad_config,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries);

// Calculate Indices of valid inputs in padded input buffer using halo kernel config's flattened local configs.
std::vector<uint32_t> input_indices_from_flattened_local_config(
    const std::vector<std::vector<uint16_t>>& flattened_local_config,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries);

// Calculate Indices of valid inputs in padded input buffer using halo kernel config's flattened remote configs.
std::vector<uint32_t> input_indices_from_flattened_remote_config(
    tt::tt_metal::IDevice* device,
    const std::vector<std::vector<uint16_t>>& flattened_remote_config,
    const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries,
    bool remote_read = false,
    bool is_block_sharded = false,
    bool transpose_mcast = false);

}  // namespace ttnn::operations::sliding_window
