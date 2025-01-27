// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include <thread>
#include <gmock/gmock.h>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/async_runtime.hpp"
#include <tt-metalium/event.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::FloatEq;
using ::testing::Pointwise;

using MultiProducerCommandQueueTest = ttnn::MultiCommandQueueSingleDeviceFixture;

TEST_F(MultiProducerCommandQueueTest, Stress) {
    // Spawn 2 application level threads intefacing with the same device through the async engine.
    // This leads to shared access of the work_executor and host side worker queue.
    // Test thread safety.
    IDevice* device = this->device_;
    // Enable async engine and set queue setting to lock_based
    device->enable_async(true);

    const ttnn::SimpleShape tensor_shape{1, 1, 1024, 1024};
    const MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    const TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    // Thread 0 uses cq_0, thread 1 uses cq_1
    const uint32_t t0_io_cq = 0;
    const uint32_t t1_io_cq = 1;

    std::vector<float> t0_host_data(tensor_shape.volume());
    std::vector<float> t1_host_data(tensor_shape.volume());
    std::iota(t0_host_data.begin(), t0_host_data.end(), 1024);
    std::iota(t1_host_data.begin(), t1_host_data.end(), 2048);

    const Tensor t0_host_tensor = Tensor::from_vector(t0_host_data, tensor_spec);
    const Tensor t1_host_tensor = Tensor::from_vector(t1_host_data, tensor_spec);

    std::thread t0([&]() {
        for (int j = 0; j < 100; j++) {
            Tensor t0_tensor = t0_host_tensor.to(device, mem_cfg, t0_io_cq);
            EXPECT_TRUE(is_tensor_on_device(t0_tensor));
            EXPECT_THAT(t0_tensor.to_vector<float>(), Pointwise(FloatEq(), t0_host_data));
        }
    });

    std::thread t1([&]() {
        for (int j = 0; j < 100; j++) {
            Tensor t1_tensor = t1_host_tensor.to(device, mem_cfg, t1_io_cq);
            EXPECT_TRUE(is_tensor_on_device(t1_tensor));
            EXPECT_THAT(t1_tensor.to_vector<float>(), Pointwise(FloatEq(), t1_host_data));
        }
    });

    t0.join();
    t1.join();
}

TEST_F(MultiProducerCommandQueueTest, EventSync) {
    // Verify that the event_synchronize API stalls the calling thread until
    // the device records the event being polled.
    // Thread 0 = writer thread. Thread 1 = reader thread.
    // Reader cannot read until writer has correctly updated a memory location.
    // Writer cannot update location until reader has picked up data.
    // Use write_event to stall reader and read_event to stall writer.
    IDevice* device = this->device_;
    // Enable async engine and set queue setting to lock_based
    device->enable_async(true);

    const ttnn::SimpleShape tensor_shape{1, 1, 1024, 1024};
    const MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};
    const TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    const uint32_t write_cq = 0;
    const uint32_t read_cq = 1;

    std::shared_ptr<Event> write_event = std::make_shared<Event>();
    std::shared_ptr<Event> read_event = std::make_shared<Event>();

    std::vector<float> host_data(tensor_shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);
    Tensor device_tensor = create_device_tensor(tensor_spec, device);

    std::thread t0([&]() {
        for (int j = 0; j < 1000; j++) {
            if (j != 0) {
                ttnn::event_synchronize(read_event);
            }
            read_event = std::make_shared<Event>();

            // Create tensor and transfer to device
            const Tensor host_tensor = Tensor::from_vector(host_data, tensor_spec);
            memcpy(device->command_queue(write_cq), device_tensor, host_tensor);
            EXPECT_TRUE(is_tensor_on_device(device_tensor));

            ttnn::record_event(device->command_queue(write_cq), write_event);
        }
    });

    std::thread t1([&]() {
        for (int j = 0; j < 1000; j++) {
            ttnn::event_synchronize(write_event);
            write_event = std::make_shared<Event>();

            // Read back from device and verify
            const Tensor readback_tensor = device_tensor.cpu(/*blocking=*/false, read_cq);
            EXPECT_FALSE(is_tensor_on_device(readback_tensor));
            EXPECT_THAT(readback_tensor.to_vector<float>(), Pointwise(FloatEq(), host_data));

            ttnn::record_event(device->command_queue(read_cq), read_event);
        }
    });

    t0.join();
    t1.join();
}

}  // namespace
}  // namespace tt::tt_metal
