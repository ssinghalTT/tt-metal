# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.bfloat16, ttnn.uint16),
        (ttnn.bfloat16, ttnn.int32),
        (ttnn.bfloat16, ttnn.uint32),
        (ttnn.bfloat8_b, ttnn.uint16),
        (ttnn.bfloat8_b, ttnn.int32),
        (ttnn.bfloat8_b, ttnn.uint32),
        (ttnn.bfloat4_b, ttnn.uint32),
        (ttnn.bfloat4_b, ttnn.uint16),
        (ttnn.bfloat4_b, ttnn.int32),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
        [1, 3, 128, 128],
    ],
)
def test_sub_with_different_dtypes(device, shape, input_dtype, output_dtype):
    x_torch = torch.ones([1, 1, 32, 32], dtype=torch.int32) * 16
    y_torch = torch.ones([1, 1, 32, 32], dtype=torch.int32) * 4
    # x_torch = torch.randint(low=1, high=100, size=shape, dtype=torch.int32)
    # y_torch = torch.randint(low=1, high=100, size=shape, dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    z_tt_sub = ttnn.experimental.sub(x_tt, y_tt, dtype=output_dtype)
    tt_out = ttnn.to_torch(z_tt_sub, dtype=torch.int32)

    print(x_tt)
    print(y_tt)
    print(f"Input dtype: {input_dtype}, Output dtype: {output_dtype}")
    print("z_torch:", z_torch)
    out = ttnn.to_torch(z_tt_sub)
    print("tt_out:", out)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status
