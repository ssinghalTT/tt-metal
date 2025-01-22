// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"
#include <tt-metalium/host_api.hpp>

#include <cstdint>

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

ReduceScatter create_reduce_scatter_struct(
    const Tensor& input_tensor,
    const ttnn::operations::binary::BinaryOpType binary_op_type,
    const uint32_t scatter_dim,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology) {
    uint32_t num_devices = devices.size();

    auto [device_index, sender_device_id, receiver_device_id] =
        get_device_index_and_sender_receiver_ids(input_tensor, devices, topology);

    TT_FATAL(
        receiver_device_id != std::nullopt || sender_device_id != std::nullopt,
        "Error, Reduce-scatter was unable to identify either a sender or receiver device ID and atleast one must be "
        "identified for a valid Reduce-scatter configuration. The input mesh tensor or Reduce-scatter arguments may be "
        "incorrect");

    return ttnn::ReduceScatter{
        binary_op_type,
        scatter_dim,
        num_links,
        num_devices,
        device_index,
        receiver_device_id,
        sender_device_id,
        output_mem_config,
        topology,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel};
}
}  // namespace reduce_scatter_detail
}  // namespace ccl

void ReduceScatter::validate(const std::vector<Tensor>& input_tensors) const {
    for (auto const& t : input_tensors) {
        TT_FATAL(
            t.get_legacy_shape()[this->scatter_dim] / this->ring_size > 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
        TT_FATAL(
            t.get_legacy_shape()[this->scatter_dim] % this->ring_size == 0,
            "Reduce scatter input tensor shape on dim {} must be divisible by ring size",
            this->scatter_dim);
    }
}

std::vector<ttnn::TensorSpec> ReduceScatter::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.get_logical_shape();
    TT_FATAL(
        shape[this->scatter_dim] % this->ring_size == 0,
        "The size of the scatter dimension must be a multiple of the ring size. Dimension size: {}, ring Size: {}",
        shape[this->scatter_dim],
        this->ring_size);
    shape[this->scatter_dim] /= this->ring_size;
    TensorSpec spec(
        shape, TensorLayout(input_tensor.get_dtype(), PageConfig(input_tensor.get_layout()), output_mem_config));
    return std::vector<ttnn::TensorSpec>(input_tensors.size(), spec);
}

operation::ProgramWithCallbacks ReduceScatter::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return ccl::reduce_scatter_detail::reduce_scatter_with_workers(
        input_tensors.at(0),
        output_tensors.at(0),
        this->binary_op_type,
        this->scatter_dim,
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->receiver_device_id,
        this->sender_device_id,
        this->topology,
        this->user_defined_num_workers,
        this->user_defined_num_buffers_per_channel);
}

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
ttnn::operations::binary::BinaryOpType convert_reduce_type_to_eltwise_type(
    ttnn::operations::reduction::ReduceType reduce_op) {
    // Leaving switch statement for future support of additional types.
    switch (reduce_op) {
        case ttnn::operations::reduction::ReduceType::Sum: return ttnn::operations::binary::BinaryOpType::ADD;
        default:
            TT_THROW("Reduce scatter only supports reduce_type Sum. Op type {} not supported.", reduce_op);
            return ttnn::operations::binary::BinaryOpType::ADD;
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace operations {
namespace ccl {
Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const MemoryConfig& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(math_op);
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "reduce_scatter op is only supported for Fast Dispatch");

    ttnn::ccl::Topology ccl_topology = topology;
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    TT_FATAL(num_devices > 1, "reduce_scatter op will only work for num_devices > 1, but has {}", num_devices);
    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    int16_t rank = input_tensor.get_logical_shape().rank();

    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        scatter_dim >= -rank && scatter_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [binary_op_type,
         scatter_dim,
         num_links,
         output_mem_config,
         ccl_topology,
         devices,
         user_defined_num_workers,
         user_defined_num_buffers_per_channel](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors.at(0);

            return operation::run(
                ttnn::ccl::reduce_scatter_detail::create_reduce_scatter_struct(
                    input_tensor,
                    binary_op_type,
                    scatter_dim,
                    num_links,
                    output_mem_config,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel,
                    devices,
                    ccl_topology),
                {input_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

Tensor reduce_scatter(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::operations::reduction::ReduceType reduce_op,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& output_mem_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    ttnn::operations::binary::BinaryOpType binary_op_type = convert_reduce_type_to_eltwise_type(reduce_op);

    TT_FATAL(
        topology == ttnn::ccl::Topology::Linear,
        "This all_gather API with cluster_axis is currently supported only for the Linear topology");
    const auto mesh_view = mesh_device.get_view();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int16_t rank = input_tensor.get_logical_shape().rank();

    int16_t scatter_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        scatter_dim >= -rank && scatter_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    operation::launch_op(
        [scatter_dim,
         binary_op_type,
         num_links,
         output_mem_config,
         mesh_view,
         cluster_axis,
         user_defined_num_workers,
         user_defined_num_buffers_per_channel,
         num_devices,
         topology](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            const auto view_index = (cluster_axis == 0) ? coordinate.col : coordinate.row;
            const auto device_index = (cluster_axis == 0) ? coordinate.row : coordinate.col;

            auto get_chip_id = [&](std::size_t line_index) -> std::optional<chip_id_t> {
                auto new_coord = coordinate;
                if (cluster_axis == 0) {
                    new_coord.row = line_index % num_devices;
                } else {
                    new_coord.col = line_index % num_devices;
                }
                return mesh_view.find_device_id(new_coord);
            };

            bool is_last_chip_in_clockwise_direction = device_index == (num_devices - 1);
            bool is_last_chip_in_counter_clockwise_direction = device_index == 0;
            auto receiver_device_id =
                is_last_chip_in_clockwise_direction ? std::nullopt : get_chip_id(device_index + 1);
            auto sender_device_id = is_last_chip_in_counter_clockwise_direction
                                        ? std::nullopt
                                        : get_chip_id(device_index + num_devices - 1);

            return operation::run(
                ttnn::ReduceScatter{
                    binary_op_type,
                    scatter_dim,
                    num_links,
                    num_devices,
                    device_index,
                    receiver_device_id,
                    sender_device_id,
                    output_mem_config.value_or(input_device_tensor.memory_config()),
                    topology,
                    user_defined_num_workers,
                    user_defined_num_buffers_per_channel},
                {input_device_tensor});
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace operations

};  // namespace ttnn
