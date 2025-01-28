// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {
namespace {

using ::testing::IsNull;
using ::testing::Not;

using MeshTensorTest = T3kMultiDeviceFixture;

TensorSpec get_tensor_spec(const ttnn::SimpleShape& shape, DataType dtype) {
    return TensorSpec(shape, TensorLayout(dtype, Layout::ROW_MAJOR, MemoryConfig{}));
}

TEST_F(MeshTensorTest, Lifecycle) {
    const TensorSpec tensor_spec =
        TensorSpec(ttnn::SimpleShape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));

    Tensor input_tensor = allocate_tensor_on_mesh(tensor_spec, mesh_device_.get());

    EXPECT_EQ(input_tensor.workers.size(), mesh_device_->num_devices());
    EXPECT_TRUE(input_tensor.is_allocated());

    const auto& storage = input_tensor.get_storage();
    auto* multi_device_storage = std::get_if<tt::tt_metal::MultiDeviceStorage>(&storage);

    ASSERT_NE(multi_device_storage, nullptr);
    EXPECT_THAT(multi_device_storage->mesh_buffer, Not(IsNull()));

    // Buffer address stays the same across all device buffers.
    const auto buffer_address = multi_device_storage->mesh_buffer->address();
    for (auto* device : mesh_device_->get_devices()) {
        auto buffer = multi_device_storage->get_buffer_for_device(device);
        ASSERT_NE(buffer, nullptr);
        EXPECT_TRUE(buffer->is_allocated());
        EXPECT_EQ(buffer->address(), buffer_address);
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
