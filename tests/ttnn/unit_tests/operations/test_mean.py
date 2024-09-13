# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, is_wormhole_b0, is_wormhole_b0


@pytest.mark.parametrize("batch_size", [1, 16, 1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
def test_mean(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    if is_wormhole_b0() and dim == -2:
        pytest.skip("Issue #6991: Wormhole B0: mean operation fails for dim=-2")

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=True, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


# mean_input_tensor:  torch.Size([1, 256, 56, 56])
# mean_input_tensor:  torch.Size([1, 512, 28, 28])
# mean_input_tensor:  torch.Size([1, 768, 14, 14])
# mean_input_tensor:  torch.Size([1, 1024, 7, 7])


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 256, 64, 64), (2, 3)),
        ((1, 512, 32, 32), (2, 3)),
        ((1, 768, 16, 16), (2, 3)),
    ],
)
def test_mean_vovnet(device, input_shape, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=True, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    print(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)
