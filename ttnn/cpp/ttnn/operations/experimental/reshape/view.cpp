// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::reshape {

Tensor tensor_reshape(
    const Tensor& input_tensor, const ttnn::SimpleShape& new_logical_shape, const ttnn::SimpleShape& new_padded_shape) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);
    auto new_spec = ttnn::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.get_dtype(),
            input_tensor.get_tensor_spec().page_config(),
            input_tensor.memory_config(),
            new_logical_shape,
            new_padded_shape));
    auto output = std::visit(
        [&input_tensor, &new_spec, &new_logical_shape, &new_padded_shape](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = input_tensor;
            if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                auto updated_storage = std::get<T>(tensor.get_storage());
                for (int i = 0; i < updated_storage.specs.size(); i++) {
                    const auto& prev_spec = updated_storage.specs[i];
                    TensorSpec spec(
                        new_logical_shape,
                        TensorLayout::fromPaddedShape(
                            prev_spec.data_type(),
                            prev_spec.page_config(),
                            prev_spec.memory_config(),
                            new_logical_shape,
                            new_padded_shape));
                    updated_storage.specs[i] = spec;
                }
                return Tensor(updated_storage, new_spec);
            }
            if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                MultiDeviceStorage updated_storage = std::get<T>(tensor.get_storage());
                std::unordered_map<int, ttnn::TensorSpec> new_specs;
                for (auto device_id : updated_storage.ordered_device_ids) {
                    const auto& prev_spec = updated_storage.specs.at(device_id);
                    TensorSpec spec(
                        new_logical_shape,
                        TensorLayout::fromPaddedShape(
                            prev_spec.data_type(),
                            prev_spec.page_config(),
                            prev_spec.memory_config(),
                            new_logical_shape,
                            new_padded_shape));
                    new_specs.insert({device_id, spec});
                }
                updated_storage.specs = new_specs;
                return Tensor(updated_storage, new_spec);
            }
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                    if (tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                        DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        DeviceBuffer device_buffer = device_storage.get_buffer();
                        const auto& tensor_spec = tensor.tensor_spec();
                        auto page_size_bytes = tensor_spec.compute_page_size_bytes();
                        device_buffer->set_page_size(page_size_bytes);
                        device_storage.insert_buffer(device_buffer);
                        return Tensor(device_storage, new_spec);
                    } else {
                        DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        DeviceBuffer device_buffer = device_storage.get_buffer();
                        ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

                        auto shard_spec = shard_spec_buffer.tensor_shard_spec;
                        auto shard_shape = shard_spec.shape;

                        uint32_t mul_div;
                        if (new_logical_shape[-1] == 0 || shard_shape[1] == 0) {
                            mul_div = 0;
                        } else {
                            mul_div = new_logical_shape[-1] > shard_shape[1] ? (new_logical_shape[-1] / shard_shape[1])
                                                                             : (shard_shape[1] / new_logical_shape[-1]);
                        }

                        shard_spec.shape[0] = new_logical_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div
                                                                                     : shard_shape[0] * mul_div;
                        shard_spec.shape[1] = new_logical_shape[-1];

                        shard_spec_buffer.page_shape = {1, new_logical_shape[-1]};
                        shard_spec_buffer.tensor2d_shape = {shard_spec.shape[0], 1};
                        shard_spec_buffer.set_shard_spec(shard_spec);

                        device_buffer->set_shard_spec(shard_spec_buffer);
                        device_storage.insert_buffer(device_buffer);

                        MemoryConfig mem_config = input_tensor.memory_config();
                        mem_config.shard_spec = shard_spec;

                        auto upd_spec = ttnn::TensorSpec(
                            new_logical_shape,
                            TensorLayout::fromPaddedShape(
                                input_tensor.get_dtype(),
                                input_tensor.get_tensor_spec().page_config(),
                                mem_config,
                                new_logical_shape,
                                new_padded_shape));

                        return Tensor(device_storage, upd_spec);
                    }
                } else {
                    return Tensor(tensor.get_storage(), new_spec);
                }
            } else {
                return Tensor(tensor.get_storage(), new_spec);
            }
        },
        input_tensor.get_storage());
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

ttnn::Tensor ViewOperation::invoke(
    const ttnn::Tensor& tensor, const ttnn::SimpleShape& logical_shape, const ttnn::SimpleShape& padded_shape) {
    return tensor_reshape(tensor, logical_shape, padded_shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::SimpleShape& shape) {
    return tensor_reshape(tensor, shape, shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tensor_reshape(tensor, shape.logical_shape(), shape.padded_shape());
}

}  // namespace ttnn::operations::experimental::reshape
