# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_blackhole


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_permute(device, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 1, 3, 2))

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_transpose(device, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_permute_on_4D_tensor_with_smaller_tuple_size(device, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    with pytest.raises(
        RuntimeError,
        match="The number of dimensions in the tensor input does not match the length of the desired ordering",
    ) as exception:
        ttnn.permute(input_tensor, (0, 1, 2))


@pytest.mark.parametrize(
    "perm", [(0,), (0, 1), (1, 0), (0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
)
def test_permute_on_less_than_4D(device, perm):
    torch.manual_seed(2005)
    tuple_shape = tuple([32 * (value + 1) for value in perm])
    torch_input_tensor = torch.rand(tuple_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.permute(torch_input_tensor, perm)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, perm)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("s", [8])
@pytest.mark.parametrize("h", [1500])
@pytest.mark.parametrize("w", [64])
def test_permute_for_specific_case(device, b, s, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((b, s, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, 1, 3, 2))
    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2))
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


def test_add_after_permute(device):
    torch.manual_seed(2005)
    torch_a = torch.randn(2, 1280, 8, 8)
    torch_b = torch.randn(1, 1, 2, 1280)
    torch_b_permuted = torch.permute(torch_b, (2, 3, 0, 1))
    torch_output = torch_a + torch_b_permuted

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.permute(b, (2, 3, 0, 1))
    output = a + b
    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_permute_negative_dim(device, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.permute(torch_input_tensor, (0, -3, -1, -2))

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.permute(input_tensor, (0, -3, -1, -2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def test_permute_bfloat8(device):
    torch.manual_seed(2005)
    input_a = torch.randn(1, 160, 32, 32)
    torch_output = torch.permute(input_a, (0, 2, 3, 1))

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_output = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape", [(8, 2, 2, 3, 4), [1, 1370, 1, 3, 1280], [1, 197, 1, 3, 1024], [1, 197, 1, 3, 768], [1, 50, 1, 3, 1024]]
)
@pytest.mark.parametrize("perm", [(0, 3, 2, 1, 4), (3, 1, 2, 0, 4), (0, 3, 2, 1, 4), (1, 3, 2, 0, 4), (0, 3, 1, 2, 4)])
def test_permute_5d(shape, perm, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("pad_value", [float("-inf"), None])
def test_permute_pad_value(device, pad_value):
    if pad_value is not None and is_blackhole():
        pytest.skip("Blackhole reduce is needed for the full test to work")
    torch.manual_seed(2005)
    input_a = torch.randn((2, 11, 33, 17), dtype=torch.bfloat16)
    torch_output = torch.permute(input_a, (3, 2, 1, 0))

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_output = ttnn.permute(tt_input, (3, 2, 1, 0), pad_value=pad_value)
    if pad_value is not None:
        a = ttnn.min(tt_output)
        assert ttnn.to_torch(a) == float("-inf")
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
