# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("n", [1])
def test_multiplyadd(device, batch_size, h, w, c, n):
    torch.manual_seed(0)
    compute_grid_size = device.compute_with_storage_grid_size()

    torch_input_tensor1 = torch.randn(n, c, h, w, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn(n, c, h, w, dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn(n, c, h, w, dtype=torch.bfloat16)

    # torch_input_tensor1 = torch.randn(batch_size, 32, 32, dtype=torch.bfloat16)
    # torch_input_tensor2 = torch.randn(batch_size, 32, 32, dtype=torch.bfloat16)
    # torch_input_tensor3 = torch.randn(batch_size, 32, 32, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.multiplyadd)
    torch_output_tensor = golden_function(torch_input_tensor1, torch_input_tensor2, torch_input_tensor3)

    tensor_memory_config = ttnn.create_sharded_memory_config(
        (n, c, h, w),
        ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # tensor_memory_config = ttnn.create_sharded_memory_config(
    #     (32, 32),
    #     ttnn.CoreGrid(y=1, x=1),
    #     strategy=ttnn.ShardStrategy.HEIGHT,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )

    input_tensor1 = ttnn.from_torch(
        torch_input_tensor1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_memory_config,
    )
    # print("memory")
    # print(tensor_memory_config)
    # print("1\n")
    # print(input_tensor1)
    # print(f"input tensor mem cfg {input_tensor1.memory_config()}")\
    input_tensor2 = ttnn.from_torch(
        torch_input_tensor2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_memory_config,
    )
    # print("2\n")
    # print(input_tensor2)
    input_tensor3 = ttnn.from_torch(
        torch_input_tensor3,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_memory_config,
    )
    # print("3\n")
    # print(input_tensor3)
    output_tensor = ttnn.multiplyadd(input_tensor1, input_tensor2, input_tensor3)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    print("output tensor")
    print(torch_output_tensor)

    assert_with_pcc(torch_output_tensor, ttnn.to_torch(output_tensor), pcc=0.99)
