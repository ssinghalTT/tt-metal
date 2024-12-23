# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from functools import partial

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [32, 48, 64, 80, 96, 112, 128])
@pytest.mark.parametrize("c", [9 * 64])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_multiplyadd(device, batch_size, h, w, c, n, dtype):
    torch.manual_seed(0)

    torch_input_tensor1 = torch.randn(batch_size, h, w, c, n, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn(batch_size, h, w, c, n, dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn(batch_size, h, w, c, n, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.multiplyadd)
    torch_output_tensor = golden_function(torch_input_tensor1, torch_input_tensor2, torch_input_tensor3)

    input_tensor1 = ttnn.from_torch(
        torch_input_tensor1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor2 = ttnn.from_torch(
        torch_input_tensor2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor3 = ttnn.from_torch(
        torch_input_tensor3,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.multiplyadd(input_tensor1, input_tensor2, input_tensor3)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    assert_with_pcc(torch_output_tensor, ttnn.to_torch(output_tensor), pcc=0.99)
