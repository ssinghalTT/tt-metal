# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from functools import reduce

layouts = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]

dtypes = [torch.float32]
shapes = [(1,), (2,), (2, 3), (2, 1, 3), (4, 3, 1, 2, 2)]
repeat_shapes = [
    (1,),
    (2,),
    (2, 1),
    (1, 2),
    (2, 1, 3),
    (1, 2, 3),
    (4, 3, 2, 1),
    (2, 3, 4, 5, 2),
    (2, 1, 3, 1, 3, 1),
]


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_repeat(device, layout, dtype, shape, repeat_shape):
    if dtype == torch.bfloat16 and shape[-1] < 2 and repeat_shape[-1] < 2:
        pytest.skip("bfloat16 needs 4 byte inner dim on the output.")

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=dtype).reshape(shape)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)
