# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [230])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize(
    "padding,torch_padding",
    [
        (((0, 1), (3, 25), (32, 32)), (32, 32, 3, 25, 0, 1)),
        (((0, 1), (3, 25), (4, 6)), (4, 6, 3, 25, 0, 1)),
        (((0, 1), (3, 25), (4, 7)), (4, 7, 3, 25, 0, 1)),  # Odd padding widths (5 and 7)
    ],
)
@pytest.mark.parametrize("value", [0])
def test_pad_rm(device, n, c, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def run_pad_rm_with_program_cache(device, n, c, h, w, padding, torch_padding, value, use_program_cache):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_rm_with_program_cache(device, n, c, h, w, padding, torch_padding, value, use_program_cache):
    for _ in range(2):
        run_pad_rm_with_program_cache(device, n, c, h, w, padding, torch_padding, value, use_program_cache)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


def run_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    n_unpadded = n
    c_unpadded = c + padding[0][1] + padding[0][0]
    h_unpadded = h + padding[1][1] + padding[1][0]

    # shard config
    num_cores_x = 8
    num_cores_y = 8
    if num_cores_y > device.core_grid.y:
        num_cores_y = device.core_grid.y
    shard_h = (n * c * h + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient, False)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    # output shard config
    num_cores_x = 8
    num_cores_y = 8
    if num_cores_y > device.core_grid.y:
        num_cores_y = device.core_grid.y
    shard_h = (n_unpadded * c_unpadded * h_unpadded + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), shard_orient, False)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    tt_output_tensor = ttnn.pad(tt_input_tensor, padding=padding, value=value, memory_config=output_mem_config)

    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert tt_output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


def to_torch_padding(padspec):
    def flatten_to_tuple(padding):
        return tuple(sum(padding, ()))

    def ttnn_pad_spec_to_padding(padspec):
        input_tensor_start = padspec["input_tensor_start"]
        pad_to_shape = padspec["pad_to_shape"]
        input_shape = padspec["input_shape"]

        padding = []
        for i in range(len(pad_to_shape)):
            this_dim_padding = (input_tensor_start[i], pad_to_shape[i] - input_shape[i] - input_tensor_start[i])
            padding.append(this_dim_padding)
        return padding

    torch_padding = flatten_to_tuple(reversed(ttnn_pad_spec_to_padding(padspec)))
    return torch_padding


@pytest.mark.parametrize(
    "input_shape, pad_to_shape, input_tensor_start, pad_value, input_sharded_memory_config_args",
    [
        [
            (1, 1, 1, 4),
            (1, 1, 1, 16),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=1), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # a reduced version of esmal's test case for UNet
            (1, 1, 4, 4),
            (1, 1, 4, 16),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=1), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # width padding across large core grid, 3 sticks per core
            (1, 1, 3 * 64, 4),
            (1, 1, 3 * 64, 16),
            (0, 0, 0, 0),
            0.0,
            {"core_grid": ttnn.CoreGrid(x=8, y=8), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # width padding across large core grid, 3 sticks per core, n300 version
            (1, 1, 3 * 8 * 7, 4),
            (1, 1, 3 * 8 * 7, 16),
            (0, 0, 0, 0),
            0.0,
            {"core_grid": ttnn.CoreGrid(x=8, y=7), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # width padding only, reduced core grid
            (1, 1, 12, 8),
            (1, 1, 12, 64),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=2, y=6), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # height and width padding, small core grid
            (1, 1, 2, 4),
            (1, 1, 4, 8),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=2), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
        [
            # borys's second test case
            (1, 2, 3, 4),
            (1, 2, 32, 32),
            (0, 0, 0, 0),
            3.0,
            {"core_grid": ttnn.CoreGrid(x=1, y=6), "strategy": ttnn.ShardStrategy.HEIGHT},
        ],
    ],
)
def test_pad_rm_sharded_stickwise(
    device, input_shape, pad_to_shape, input_tensor_start, pad_value, input_sharded_memory_config_args
):
    core_grid_x_ok = device.core_grid.x >= input_sharded_memory_config_args["core_grid"].x
    core_grid_y_ok = device.core_grid.y >= input_sharded_memory_config_args["core_grid"].y
    device_core_grid_ok = core_grid_x_ok and core_grid_y_ok
    if not device_core_grid_ok:
        pytest.skip("core grid for this test is not compatible with the device")

    input_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)

    torch_input_tensor = torch.ones(input_shape, dtype=torch.float32)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    # Still relay on keep_l1_aligned = True to make it work with the current implementation
    ttnn_sharded_input_tensor = ttnn.interleaved_to_sharded(
        ttnn_input_tensor, input_shard_memory_config, keep_l1_aligned=True
    )
    padded_tensor = ttnn.pad(ttnn_sharded_input_tensor, pad_to_shape, input_tensor_start, pad_value)

    tt_output_tensor = ttnn.to_memory_config(padded_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    padspec = {
        "input_shape": input_shape,
        "pad_to_shape": pad_to_shape,
        "input_tensor_start": input_tensor_start,
    }
    torch_padded_tensor = torch.nn.functional.pad(
        torch_input_tensor, to_torch_padding(padspec), mode="constant", value=pad_value
    )

    assert torch_output_tensor.shape == torch_padded_tensor.shape
    assert_with_pcc(torch_padded_tensor, torch_output_tensor, 0.99)


@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("padding,torch_padding", [(((1, 1), (2, 32), (0, 0)), (0, 0, 2, 32, 1, 1))])
@pytest.mark.parametrize("value", [8])
@pytest.mark.parametrize("shard_orient", [ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.ROW_MAJOR])
def test_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient, use_program_cache):
    if device.core_grid.y < 8:
        pytest.skip("n300 does not have 8x8 grid")
    for _ in range(2):
        run_pad_rm_sharded(device, n, c, h, w, padding, torch_padding, value, shard_orient)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 3


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 64),), (0, 64)), (((16, 16), (0, 32)), (0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    output_tensor = ttnn.to_torch(output_tensor)
    assert output_tensor.shape == torch_output_tensor.shape

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [2, 30])
@pytest.mark.parametrize("w", [128, 60])
@pytest.mark.parametrize("padding", [((0, 32), (0, 32)), ((0, 32), (0, 64))])
@pytest.mark.parametrize("value", [0])
def test_pad_any_input_shape(device, h, w, padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    output_tensor = ttnn.to_torch(output_tensor)
    tilezed_input_shape = input_tensor.shape.with_tile_padding()
    th = tilezed_input_shape[-2]
    tw = tilezed_input_shape[-1]
    assert output_tensor.shape == ttnn.Shape((th + padding[0][0] + padding[0][1], tw + padding[1][0] + padding[1][1]))


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((32, 32),), (32, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_front_pad_not_supported(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        ttnn.pad(input_tensor, padding=padding, value=value)
    assert "ttnn.pad: on device tile padding does not support front padding" in str(e.value)
    return


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 32), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_length(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        ttnn.pad(input_tensor, padding=padding, value=value)
    assert "ttnn.pad: padding len can't be larger than input tensor rank" in str(e.value)
    return


@pytest.mark.skip(reason="ttnn.pad does not support row_major tensors because the kernel currently causes a PCC error")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.skip(reason="ttnn.pad does not support row_major tensors because the kernel currently causes a PCC error")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_back_to_back(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.pad(output_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape(
        (h + (padding[0][0] + padding[0][1]) * 2, w + (padding[1][0] + padding[1][1]) * 2)
    )

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.skip(reason="ttnn.pad requires pad to start at 0")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding", [((0, 32), (0, 32)), ((1, 64), (0, 96)), ((0, 64), (0, 43)), ((32, 64), (64, 96))])
@pytest.mark.parametrize("value", [0])
def test_pad_for_tensor_in_tile_layout(device, h, w, padding, value):
    torch.manual_seed(0)
    torch_padding = (padding[1][0], padding[1][1], padding[0][0], padding[0][1])

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    if (
        padding[0][0] % ttnn.TILE_SIZE != 0
        or padding[0][1] % ttnn.TILE_SIZE != 0
        or padding[1][0] % ttnn.TILE_SIZE != 0
        or padding[1][1] % ttnn.TILE_SIZE != 0
    ):
        with pytest.raises(RuntimeError) as e:
            output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
        assert "must be a multiple of the tile size on height and width" in str(e.value)
        return
    else:
        output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
