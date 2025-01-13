# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32 * 64])  # number of cores that does the work
@pytest.mark.parametrize("w", [32 * 2])  # shard width, number of tiles per core
def test_multiplyadd(device, h, w):
    torch.manual_seed(0)
    compute_grid_size = device.compute_with_storage_grid_size()

    torch_input_tensor1 = torch.randn((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn((h, w), dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn((h, w), dtype=torch.bfloat16)

    tensor_memory_config = ttnn.create_sharded_memory_config(
        (h, w),
        ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_function = ttnn.get_golden_function(ttnn.multiplyadd)
    torch_output_tensor = golden_function(torch_input_tensor1, torch_input_tensor2, torch_input_tensor3)

    input_tensor1 = ttnn.from_torch(
        torch_input_tensor1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_memory_config,
    )

    input_tensor2 = ttnn.from_torch(
        torch_input_tensor2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_memory_config,
    )

    input_tensor3 = ttnn.from_torch(
        torch_input_tensor3,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_memory_config,
    )

    output_tensor = ttnn.multiplyadd(input_tensor1, input_tensor2, input_tensor3)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
