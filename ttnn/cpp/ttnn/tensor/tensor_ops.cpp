// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_ops.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/core.hpp"

#include "cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace tt::tt_metal::tensor_ops {

Tensor tensor_to(const Tensor& input_tensor, IDevice* target_device, const MemoryConfig& mem_config, uint8_t cq_id) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, target_device, mem_config);
    // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
    Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(target_device, input_tensor);
    // Populate device storage outside of thread, so that downstream
    // functions running in main can get storage type without blocking
    Tensor device_tensor({target_device});
    // Record main thread ref count for tensors before pushing to queue.
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = async_safe_tensor.tensor_attributes->record_main_thread_ref_count();
    target_device->push_work([async_safe_tensor, device_tensor, mem_config, target_device, cq_id]() mutable {
        if (async_safe_tensor.storage_type() == StorageType::DEVICE) {
            TT_ASSERT(async_safe_tensor.device() == target_device && "Currently do not support moving between devices");
            device_tensor.populate_buffers_and_metadata(async_safe_tensor);
        } else {
            tensor_impl::validate_on_device_dtype_and_layout(
                target_device,
                async_safe_tensor.get_padded_shape(),
                async_safe_tensor.get_dtype(),
                async_safe_tensor.get_layout());
            auto local_tensor = tensor_impl::to_device_wrapper(async_safe_tensor, target_device, mem_config, cq_id);
            // Populate device tensor
            device_tensor.populate_buffers_and_metadata(local_tensor);
        }
    });
    // Update main thread ref count for tensors after pushing to queue (update original tensor and returned tensor,
    // since both can be on device).
    device_tensor.tensor_attributes->update_main_thread_ref_count(device_tensor.workers.at(0), device_tensor_ref_count);
    async_safe_tensor.tensor_attributes->update_main_thread_ref_count(
        device_tensor.workers.at(0), original_tensor_ref_count);
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

Tensor tensor_to(
    const Tensor& input_tensor, const std::vector<IDevice*>& workers, const MemoryConfig& mem_config, uint8_t cq_id) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, workers, mem_config);
    TT_FATAL(
        validate_worker_modes(workers), "All device threads/workers must be running in the same mode (ASYNC or SYNC)");
    Tensor device_tensor = Tensor(workers);
    uint32_t device_tensor_ref_count = device_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t original_tensor_ref_count = input_tensor.tensor_attributes->record_main_thread_ref_count();
    uint32_t num_workers = workers.size();
    for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
        auto& worker = workers[worker_index];
        worker->push_work(
            [worker, input_tensor, device_tensor, mem_config, num_workers, worker_index, cq_id]() mutable {
                auto shard = get_shard_for_device(input_tensor, worker, worker_index);
                if (shard.storage_type() == StorageType::OWNED) {
                    shard = tensor_impl::to_device_wrapper(shard, worker, mem_config, cq_id);
                }
                insert_buffer_and_shape_for_device(worker, shard, device_tensor, worker_index);
                uint32_t num_workers_completed = (device_tensor.tensor_attributes->num_workers_completed)++;
                if (not num_workers_completed) {
                    device_tensor.set_tensor_spec(TensorSpec(
                        input_tensor.get_logical_shape(),
                        input_tensor.get_tensor_spec().tensor_layout().with_memory_config(mem_config)));
                }
            });
    }
    device_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), device_tensor_ref_count);
    input_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    device_tensor = tt::tt_metal::set_tensor_id(device_tensor);
    GraphTracker::instance().track_function_end(device_tensor);
    return device_tensor;
}

Tensor tensor_cpu(const Tensor& input_tensor, bool blocking, uint8_t cq_id) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::cpu", input_tensor, blocking);
    auto workers = input_tensor.get_workers(blocking);
    if (not workers.size()) {
        // Tensor is on host and does not have a worker group.
        // Return immediately. If this is a result of .cpu() called twice,
        // tensor accessors will stall until tensor is populated.
        auto output = tt::tt_metal::set_tensor_id(input_tensor);
        GraphTracker::instance().track_function_end(output);
        return output;
    }
    TT_FATAL(
        validate_worker_modes(workers), "All device threads/workers must be running in the same mode (ASYNC or SYNC)");
    Tensor host_tensor(workers.size());
    uint32_t original_tensor_ref_count = input_tensor.tensor_attributes->record_main_thread_ref_count();
    for (int worker_index = 0; worker_index < workers.size(); worker_index++) {
        auto target_device = workers[worker_index];
        target_device->push_work(
            [host_tensor, blocking, target_device, input_tensor, worker_index, cq_id]() mutable {
                TT_ASSERT(
                    input_tensor.storage_type() == StorageType::DEVICE or
                        input_tensor.storage_type() == StorageType::MULTI_DEVICE,
                    "Can only use worker queue for cpu call if tensor is on device.");
                auto shard = get_shard_for_device(input_tensor, target_device);
                shard = tensor_impl::to_host_wrapper(shard, blocking, cq_id);
                insert_buffer_and_shape_for_device(target_device, shard, host_tensor, worker_index);
                uint32_t num_workers_completed = (host_tensor.tensor_attributes->num_workers_completed)++;
                if (not num_workers_completed) {
                    host_tensor.set_tensor_spec(input_tensor.get_tensor_spec());
                }
            });
    }

    if (blocking) {
        tt::tt_metal::detail::SynchronizeWorkerThreads(workers);
    }
    // Update main_thread_ref_count for tensor after pushing to queue.
    input_tensor.tensor_attributes->update_main_thread_ref_count(workers.at(0), original_tensor_ref_count);
    host_tensor = tt::tt_metal::set_tensor_id(host_tensor);
    GraphTracker::instance().track_function_end(host_tensor);
    return host_tensor;
}

Tensor tensor_to(const Tensor& input_tensor, Layout target_layout, IDevice* worker) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, target_layout, worker);
    // Only push layout conversion to worker if running in async mode
    if (worker and worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS) {
        // Tensor can be using borrowed storage. If so, when running in async mode, copy this tensor to owned storage.
        Tensor async_safe_tensor = copy_borrowed_tensor_in_async_mode(worker, input_tensor);
        Tensor tensor_modified_layout = Tensor(1);
        worker->push_work([async_safe_tensor, tensor_modified_layout, target_layout]() mutable {
            TT_ASSERT(
                async_safe_tensor.storage_type() == StorageType::OWNED or
                async_safe_tensor.storage_type() == StorageType::BORROWED &&
                    "to(layout) must be called on host tensors with a single buffer when a single worker is specified");
            auto local_tensor = tensor_impl::to_layout_wrapper(async_safe_tensor, target_layout);
            // Populate modified layout tensor
            tensor_modified_layout.populate_buffers_and_metadata(local_tensor);
        });
        tensor_modified_layout = tt::tt_metal::set_tensor_id(tensor_modified_layout);
        GraphTracker::instance().track_function_end(tensor_modified_layout);
        return tensor_modified_layout;
    }
    // Running without worker threads (non-async)
    TT_ASSERT(
        input_tensor.storage_type() != StorageType::DEVICE or
        input_tensor.storage_type() != StorageType::MULTI_DEVICE &&
            "Bring tensor to host before converting to target layout");
    auto output = tensor_impl::to_layout_wrapper(input_tensor, target_layout);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_to(const Tensor& input_tensor, Layout target_layout, distributed::MeshDevice* mesh_device) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::to", input_tensor, target_layout, mesh_device);
    if (mesh_device) {
        auto workers = ttnn::distributed::get_mapped_devices(input_tensor, *mesh_device);
        TT_FATAL(
            validate_worker_modes(workers),
            "All device threads/workers must be running in the same mode (ASYNC or SYNC)");

        std::optional<DistributedTensorConfig> distributed_config = std::nullopt;
        if (auto* host_storage = std::get_if<MultiDeviceHostStorage>(&input_tensor.get_storage());
            host_storage != nullptr) {
            distributed_config = host_storage->strategy;
        }

        Tensor tensor_modified_layout = Tensor(workers.size(), distributed_config);
        for (int worker_index = 0; worker_index < workers.size(); ++worker_index) {
            auto& worker = workers[worker_index];
            worker->push_work([input_tensor, tensor_modified_layout, target_layout, worker, worker_index]() mutable {
                TT_ASSERT(
                    input_tensor.storage_type() == StorageType::OWNED ||
                    input_tensor.storage_type() == StorageType::BORROWED ||
                    input_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST &&
                        "to(layout) must be called on host tensors with MULTI_DEVICE_HOST_STORAGE when multiple "
                        "workers "
                        "are specified");
                ;
                auto shard = get_shard_for_device(input_tensor, worker, worker_index);
                shard = tensor_impl::to_layout_wrapper(shard, target_layout);
                insert_buffer_and_shape_for_device(worker, shard, tensor_modified_layout, worker_index);
                uint32_t num_workers_completed = (tensor_modified_layout.tensor_attributes->num_workers_completed)++;
                if (not num_workers_completed) {
                    auto orig_layout = input_tensor.get_tensor_spec().tensor_layout();
                    auto upd_layout = TensorLayout(
                        orig_layout.get_data_type(), PageConfig(target_layout), orig_layout.get_memory_config());
                    tensor_modified_layout.set_tensor_spec(TensorSpec(input_tensor.get_logical_shape(), upd_layout));
                }
            });
        }
        tensor_modified_layout = tt::tt_metal::set_tensor_id(tensor_modified_layout);
        GraphTracker::instance().track_function_end(tensor_modified_layout);
        return tensor_modified_layout;
    }
    // Running without worker threads (non-async)
    TT_ASSERT(
        input_tensor.storage_type() != StorageType::DEVICE or
        input_tensor.storage_type() != StorageType::MULTI_DEVICE &&
            "Bring tensor to host before converting to target layout");
    auto output = tensor_impl::to_layout_wrapper(input_tensor, target_layout);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

void tensor_print(const Tensor& input_tensor) {
    GraphTracker::instance().track_function_start("Tensor::print", input_tensor);
    std::cout << input_tensor.write_to_string() << std::endl;
    GraphTracker::instance().track_function_end();
}

Tensor tensor_pad(
    const Tensor& input_tensor,
    const ttnn::SimpleShape& output_padded_shape,
    const ttnn::SimpleShape& input_tensor_start,
    float pad_value) {
    ZoneScoped;
    GraphTracker::instance().track_function_start(
        "Tensor::pad", input_tensor, output_padded_shape, input_tensor_start, pad_value);
    TT_ASSERT(
        input_tensor.storage_type() == StorageType::OWNED or
        input_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST or
        input_tensor.storage_type() == StorageType::BORROWED && "Tensor must be on host for padding");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (input_tensor.get_layout() != Layout::ROW_MAJOR) {
        log_warning(
            tt::LogOp,
            "Tensor layout {} must be ROW_MAJOR for padding! Returning original tensor!",
            input_tensor.get_layout());
        return input_tensor;
    }

    auto output = tensor_impl::pad_wrapper(input_tensor, output_padded_shape, input_tensor_start, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_unpad(
    const Tensor& input_tensor,
    const ttnn::SimpleShape& output_tensor_start,
    const ttnn::SimpleShape& output_tensor_end) {
    ZoneScoped;
    GraphTracker::instance().track_function_start(
        "Tensor::unpad", input_tensor, output_tensor_start, output_tensor_end);
    TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR && "Tensor layout must be ROW_MAJOR for unpadding");
    auto output = tensor_impl::unpad_wrapper(input_tensor, output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_pad_to_tile(const Tensor& input_tensor, float pad_value) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::pad_to_tile", input_tensor, pad_value);
    uint32_t height = input_tensor.get_padded_shape()[-2];
    uint32_t width = input_tensor.get_padded_shape()[-1];
    uint32_t padded_height = round_up(height, constants::TILE_HEIGHT);
    uint32_t padded_width = round_up(width, constants::TILE_WIDTH);

    ttnn::SmallVector<uint32_t> padded_shape;
    ttnn::SmallVector<uint32_t> input_tensor_start;

    for (auto index = 0; index < input_tensor.get_padded_shape().rank() - 2; index++) {
        padded_shape.push_back(input_tensor.get_padded_shape()[index]);
        input_tensor_start.push_back(0);
    }

    padded_shape.push_back(padded_height);
    padded_shape.push_back(padded_width);
    input_tensor_start.push_back(0);
    input_tensor_start.push_back(0);

    auto output = input_tensor.pad(
        ttnn::SimpleShape(std::move(padded_shape)), ttnn::SimpleShape{std::move(input_tensor_start)}, pad_value);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_unpad_from_tile(const Tensor& input_tensor, const ttnn::SimpleShape& output_tensor_shape) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("Tensor::unpad_from_tile", input_tensor, output_tensor_shape);

    for (auto index = -3; index >= -static_cast<int>(input_tensor.get_padded_shape().rank()); index--) {
        TT_ASSERT(
            input_tensor.get_legacy_shape().without_padding()[index] == output_tensor_shape[index],
            "Input shape must match output shape apart from last 2 dims");
    }
    TT_ASSERT(
        input_tensor.get_padded_shape()[-2] % constants::TILE_HEIGHT == 0 &&
            input_tensor.get_padded_shape()[-1] % constants::TILE_WIDTH == 0,
        "Last 2 dims of input shape must be multiples of 32");
    TT_ASSERT(
        input_tensor.get_padded_shape()[-2] < output_tensor_shape[-2] + constants::TILE_HEIGHT &&
            input_tensor.get_padded_shape()[-1] < output_tensor_shape[-1] + constants::TILE_WIDTH,
        "Last 2 dims of output must be within range to have been padded to input");
    SimpleShape output_tensor_start(ttnn::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 0));
    SimpleShape output_tensor_end(ttnn::SmallVector<uint32_t>(input_tensor.padded_shape().rank(), 1));
    for (int index = -1; index >= -static_cast<int>(output_tensor_shape.rank()); index--) {
        output_tensor_end[index] = output_tensor_shape[index];
    }
    auto output = input_tensor.unpad(output_tensor_start, output_tensor_end);
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

Tensor tensor_reshape(const Tensor& input_tensor, const ttnn::Shape& new_shape) {
    return ttnn::reshape(input_tensor, new_shape);
}

Tensor tensor_reshape(const Tensor& input_tensor, const ttnn::SimpleShape& new_shape) {
    return ttnn::reshape(input_tensor, new_shape);
}

}  // namespace tt::tt_metal::tensor_ops
