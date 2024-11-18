# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shapes",
    [
        #    [[1, 1, 64, 64], [1, 1, 64, 64]],
        [
            [4, 2, 96, 192],
        ],
    ],
)
def test_i0_range(device, shapes):
    torch.manual_seed(3624344)
    high = 1
    low = -1
    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16) * (high - low) + low
    torch_output_tensor = torch.i0(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.998)
    print("pcc_msg", pcc_msg)
    assert pcc


@pytest.mark.parametrize(
    "shapes",
    [
        #    [[1, 1, 64, 64], [1, 1, 64, 64]],
        [
            [4, 2, 96, 192],
        ],
    ],
)
def test_i1_range(device, shapes):
    torch.manual_seed(3624344)
    high = 1
    low = -1
    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i1(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.sign(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
    print("pcc_msg", pcc_msg)
    assert pcc


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [4, 7, 21, 133],
        [4, 6, 105, 245],
    ],
)
def test_i0_bw_range(device, shapes):
    torch.manual_seed(3624344)  # 16305027
    high = -10
    low = 10
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32, requires_grad=True) * (high - low) + low

    high = 5
    low = -5
    grad_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low

    golden_fn = ttnn.get_golden_function(ttnn.i0_bw)
    torch_output_tensor = golden_fn(grad_tensor_a, torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grad_tensor = ttnn.from_torch(
        grad_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0_bw(grad_tensor, input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor[0])
    torch_output_tensor = torch_output_tensor[0]

    # for i in range(shapes[0]):            # Batch size
    #     for j in range(shapes[1]):        # Channels
    #         for k in range(shapes[2]):   # Height
    #             for l in range(shapes[3]):  # Width
    #                 print(f"{i}-{j}-{k}-{l} input: {torch_input_tensor_a[i][j][k][l]} \t TT_out: {output_tensor[i][j][k][l]} \t torch: {torch_output_tensor[i][j][k][l]}\n")

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
    print("pcc_msg", pcc_msg)  # AssertionError: 0.9920072418577158
    assert pcc


def execute_unary_backward_i0(grad, input):
    # # Reciprocal of input
    # input_recip = torch.reciprocal(input)
    # handled_recip = torch.where(input == 0, torch.tensor(0.0, dtype=input.dtype), input_recip)

    # # i0(input) computation (use torch.special.i0 for the modified Bessel function of the first kind)
    # i0_input = torch.special.i0(input)

    # # Compute value = 0.5 * (i0(input) * handled_recip)
    # value = 0.5 * (i0_input * handled_recip)

    # # Negative of i0(input)
    # neg_i0_input = -i0_input

    # # Compute result
    # result = torch.where(
    #     input < 0,
    #     grad * (neg_i0_input - value),
    #     grad * (i0_input - value)
    # )

    # # Handle the case where input == 0
    # result = torch.where(input == 0, torch.tensor(0.0, dtype=input.dtype), result)

    result = grad * torch.special.i1(input)

    return result


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [4, 7, 21, 133],
        [4, 6, 105, 245],
    ],
)
def test_i0_bw_recip_range(device, shapes):
    torch.manual_seed(16305027)
    high = -10
    low = 10
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32, requires_grad=True) * (high - low) + low

    high = 5
    low = -5
    grad_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low

    # golden_fn = ttnn.get_golden_function(ttnn.i0_bw)
    torch_output_tensor = execute_unary_backward_i0(grad_tensor_a, torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grad_tensor = ttnn.from_torch(
        grad_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0_bw(grad_tensor, input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor[0])

    # for i in range(shapes[0]):            # Batch size
    #     for j in range(shapes[1]):        # Channels
    #         for k in range(shapes[2]):   # Height
    #             for l in range(shapes[3]):  # Width
    #                 print(f"{i}-{j}-{k}-{l} input: {torch_input_tensor_a[i][j][k][l]} \t TT_out: {output_tensor[i][j][k][l]} \t torch: {torch_output_tensor[i][j][k][l]}\n")

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
    print("pcc_msg", pcc_msg)
    assert pcc
